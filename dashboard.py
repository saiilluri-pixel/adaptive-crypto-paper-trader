"""
Live A/B comparison dashboard. Stdlib only.

    python3 dashboard.py            # http://localhost:8787

Reads the combined state.json (nested by variant) plus per (variant,symbol)
trades_<V>_<SYM>.csv / equity_<V>_<SYM>.csv. Page at / auto-refreshes; JSON at
/api/state.
"""
import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def _read_json(name):
    try:
        with open(os.path.join(HERE, name)) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_trades(fname, n=30):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-n:][::-1]


def _read_equity(fname):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                t, v = line.strip().split(",")
                out.append([int(t), float(v)])
            except Exception:
                pass
    return out[-720:]


def _max_dd(curve):
    peak, mdd = -1e18, 0.0
    for _, v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1)
    return mdd * 100


def build_api():
    st = _read_json("state.json")
    all_trades = []
    for vk, vd in st.get("variants", {}).items():
        for sym, sd in vd.get("symbols", {}).items():
            ftag = f"{vk}_{sym.replace('/', '')}"
            curve = _read_equity(f"equity_{ftag}.csv")
            sd["equity_curve"] = curve
            sd["max_dd"] = _max_dd(curve) if curve else 0.0
            trades = _read_trades(f"trades_{ftag}.csv")
            sd["trades"] = trades
            tag = sym.split("/")[0]
            for tr in trades:
                t2 = dict(tr); t2["_sym"] = tag; t2["_var"] = vk
                all_trades.append(t2)
    st["all_trades"] = sorted(all_trades, key=lambda t: t.get("exit_time", ""),
                              reverse=True)[:50]
    return st


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B Paper Trade</title>
<style>
:root{--bg:#0a0e14;--panel:#131722;--panel2:#1c2230;--line:#2a2e39;--tx:#d1d4dc;
--mut:#787b86;--grn:#26a69a;--red:#ef5350;--yel:#f7b500;--blu:#4f7cff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font:14px/1.4 -apple-system,Segoe UI,Roboto,monospace}
.wrap{max-width:1320px;margin:0 auto;padding:18px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px}
header h1{font-size:17px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;background:var(--grn);box-shadow:0 0 8px var(--grn);animation:p 1.6s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
.upd{color:var(--mut);font-size:12px;margin-left:auto}
.h2h{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:18px}
@media(max-width:760px){.h2h{grid-template-columns:1fr}}
.vsum{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.vsum.win{border-color:var(--grn);box-shadow:0 0 0 1px var(--grn)}
.vsum .vh{display:flex;align-items:baseline;gap:8px;margin-bottom:8px}
.vsum .vk{font-size:13px;font-weight:700;padding:1px 8px;border-radius:5px;background:var(--panel2);color:var(--blu)}
.vsum .vl{font-size:12.5px;color:var(--mut)}
.vsum .big{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}
.vsum .sub{font-size:12px;color:var(--mut);margin-top:2px}
.lead{margin-left:auto;font-size:11px;font-weight:700;color:var(--grn);align-self:center}
.section h2{font-size:13px;margin:6px 0 10px;font-weight:600;display:flex;gap:8px;align-items:center}
.section h2 .vk{font-size:11px;padding:1px 7px;border-radius:5px;background:var(--panel2);color:var(--blu)}
.syms{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.chead{display:flex;align-items:baseline;gap:9px;margin-bottom:10px}
.coin{font-size:17px;font-weight:700}.cpx{font-size:14px;color:var(--mut);font-variant-numeric:tabular-nums}
.cret{margin-left:auto;font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
.mini{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px}
.mini .k{color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.4px}
.mini .v{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:700;font-size:11px}
.long{background:rgba(38,166,154,.15);color:var(--grn);border:1px solid var(--grn)}
.short{background:rgba(239,83,80,.15);color:var(--red);border:1px solid var(--red)}
.flat{background:var(--panel2);color:var(--mut);border:1px solid var(--line)}
.pos{color:var(--grn)}.neg{color:var(--red)}.muted{color:var(--mut)}
.posline{font-size:12px;margin:8px 0;line-height:1.5}
.row2{display:grid;grid-template-columns:1fr 132px;gap:10px;align-items:start}
.ladder{position:relative;height:140px;border-left:2px solid var(--line)}
.lv{position:absolute;left:0;right:0;display:flex;align-items:center;gap:5px;transform:translateY(-50%);font-size:10.5px;white-space:nowrap}
.lv .tick{width:9px;height:2px}.lv .px{margin-left:auto;font-variant-numeric:tabular-nums}
.pxmark{position:absolute;left:0;right:0;height:0;border-top:2px dashed var(--yel)}
.pxmark span{position:absolute;right:0;top:-8px;background:var(--yel);color:#000;font-weight:700;font-size:10px;padding:0 5px;border-radius:3px}
svg{display:block;width:100%;height:42px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);font-size:12px;white-space:nowrap}
th{color:var(--mut);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.4px}
th:first-child,td:first-child{text-align:left}
.box{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}
.box h2{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;font-weight:600}
.vchip{font-weight:700;padding:0 6px;border-radius:4px;background:var(--panel2);color:var(--blu);font-size:11px}
</style></head>
<body><div class="wrap">
<header><span class="dot"></span>
  <h1>A/B paper trade <span class="muted" style="font-weight:400">· Prime ⊕ Supertrend · live</span></h1>
  <span class="upd" id="upd">connecting…</span></header>

<div class="h2h" id="h2h"></div>
<div id="sections"></div>
<div class="box"><h2>Recent trades (both variants)</h2>
  <table><thead><tr><th>Exit (UTC)</th><th>Var</th><th>Coin</th><th>Side</th><th>Entry</th>
  <th>Exit</th><th>Net P&amp;L</th><th>Reason</th></tr></thead><tbody id="trades"></tbody></table>
</div>
</div>

<script>
const fmt=n=>n.toLocaleString(undefined,{maximumFractionDigits:n<100?2:0});
const fmt2=n=>n.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const money=n=>(n<0?'-':'')+'$'+Math.abs(n).toLocaleString(undefined,{maximumFractionDigits:2});
const cls=n=>n>0?'pos':n<0?'neg':'';
function utcClock(s){
  if(!s) return '';
  let iso=s.includes('T')?s:s.replace(' ','T')+(/[+Z]/.test(s)?'':'Z');
  const d=new Date(iso);
  return isNaN(d)?s:d.toLocaleString([], {timeZone:'UTC',month:'2-digit',day:'2-digit',
    hour:'2-digit',minute:'2-digit',hour12:false});
}

async function tick(){
  let d; try{ d=await (await fetch('/api/state',{cache:'no-store'})).json(); }
  catch(e){ document.getElementById('upd').textContent='no data yet'; return; }
  const V=d.variants; if(!V){ document.getElementById('upd').textContent='waiting for bot…'; return; }
  document.getElementById('upd').textContent='updated '+utcClock(d.ts)+' UTC';

  const keys=Object.keys(V);
  const rets=keys.map(k=>V[k].totals.return_pct||0);
  const best=rets.indexOf(Math.max(...rets));

  document.getElementById('h2h').innerHTML=keys.map((k,i)=>{
    const t=V[k].totals, win=(i===best && keys.length>1 && rets[best]!==Math.min(...rets));
    return `<div class="vsum ${win?'win':''}">
      <div class="vh"><span class="vk">${k}</span><span class="vl">${V[k].label}</span>
        ${win?'<span class="lead">▲ leading</span>':''}</div>
      <div class="big ${cls(t.return_pct)}">${t.return_pct>=0?'+':''}${(t.return_pct||0).toFixed(2)}%</div>
      <div class="sub">equity ${money(t.equity)} · realized ${money(t.realized||0)} · ${t.n_trades||0} closed trades</div>
    </div>`;
  }).join('');

  document.getElementById('sections').innerHTML=keys.map(k=>{
    const syms=V[k].symbols;
    return `<div class="section"><h2><span class="vk">${k}</span> ${V[k].label}</h2>
      <div class="syms">${Object.keys(syms).map(s=>symCard(k,s,syms[s])).join('')}</div></div>`;
  }).join('');
  keys.forEach(k=>Object.keys(V[k].symbols).forEach(s=>{
    drawLadder(k,s,V[k].symbols[s]); drawCurve(k,s,V[k].symbols[s]);
  }));

  document.getElementById('trades').innerHTML=(d.all_trades||[]).map(tr=>{
    const net=parseFloat(tr.net_pnl);
    return `<tr><td>${utcClock(tr.exit_time)}</td><td><span class="vchip">${tr._var}</span></td>
      <td>${tr._sym}</td><td class="${tr.side==='long'?'pos':'neg'}">${tr.side.toUpperCase()}</td>
      <td>$${fmt(+tr.entry_px)}</td><td>$${fmt(+tr.exit_px)}</td>
      <td class="${net>=0?'pos':'neg'}">${money(net)}</td><td class="muted">${tr.exit_reason}</td></tr>`;
  }).join('') || '<tr><td colspan="8" class="muted">no closed trades yet</td></tr>';
}

function symCard(vk, sym, sd){
  const tag=sym.split('/')[0], m=sd.market||{}, p=sd.position, ret=sd.return_pct||0;
  const wr=sd.n_trades?Math.round(sd.wins/sd.n_trades*100):0;
  let posHtml;
  if(p){
    const c=p.side==='long'?'long':'short';
    posHtml=`<div class="posline"><span class="badge ${c}">${p.side.toUpperCase()}</span> `+
      `${p.qty.toFixed(4)} @ $${fmt(p.entry)} · stop $${fmt(p.stop)} · `+
      `<span class="${cls(p.unrealized)}">${money(p.unrealized)}</span></div>`;
  } else {
    let w='waiting for signal';
    if(m.fz_trend===-1&&m.z70) w=`BEAR · short ≥ $${fmt(m.z70)}`;
    else if(m.fz_trend===1&&m.z30) w=`BULL · long ≤ $${fmt(m.z30)}`;
    posHtml=`<div class="posline"><span class="badge flat">FLAT</span> <span class="muted">${w}</span></div>`;
  }
  const trB=m.fz_trend===1?'<span class="badge long">BULL</span>':m.fz_trend===-1?'<span class="badge short">BEAR</span>':'<span class="badge flat">—</span>';
  const stB=m.st_dir===1?'<span class="badge long">ST UP</span>':'<span class="badge short">ST DOWN</span>';
  return `<div class="card">
    <div class="chead"><span class="coin">${tag}</span><span class="cpx">$${fmt2(sd.price)}</span>
      <span class="cret ${cls(ret)}">${ret>=0?'+':''}${ret.toFixed(2)}%</span></div>
    <div class="mini">
      <div><div class="k">Equity</div><div class="v">${money(sd.equity)}</div></div>
      <div><div class="k">Trades</div><div class="v">${sd.n_trades} <span class="muted" style="font-size:10px">(${wr}%)</span></div></div>
      <div><div class="k">Max DD</div><div class="v">${(sd.max_dd||0).toFixed(1)}%</div></div>
    </div>
    <svg id="cv_${vk}_${tag}" viewBox="0 0 600 42" preserveAspectRatio="none"></svg>
    ${posHtml}
    <div style="display:flex;gap:6px;margin:6px 0 4px">${trB}${stB}</div>
    <div class="row2"><div style="font-size:10.5px;color:var(--mut);padding-top:22px">zones →</div>
      <div class="ladder" id="lad_${vk}_${tag}"></div></div>
  </div>`;
}

function drawLadder(vk, sym, sd){
  const tag=sym.split('/')[0], m=sd.market||{}, el=document.getElementById('lad_'+vk+'_'+tag);
  if(!el) return;
  if(!m.z15){el.innerHTML='<div class="muted" style="padding-top:55px;font-size:10.5px">zone not set</div>';return}
  const L=[['85%s',m.z85,'var(--red)',true],['70%s',m.z70,'var(--red)',m.fz_trend===-1],
           ['70%b',m.z30,'var(--grn)',m.fz_trend===1],['85%b',m.z15,'var(--grn)',true]];
  const price=sd.price, vals=L.map(x=>x[1]).concat([price]);
  let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo)*0.10||1;lo-=pad;hi+=pad;
  const y=v=>(1-(v-lo)/(hi-lo))*100;
  let h=L.map(x=>`<div class="lv" style="top:${y(x[1]).toFixed(1)}%;opacity:${x[3]?1:0.4}">
     <span class="tick" style="background:${x[2]}"></span><span class="muted">${x[0]}</span>
     <span class="px">$${fmt(x[1])}</span></div>`).join('');
  h+=`<div class="pxmark" style="top:${y(price).toFixed(1)}%"><span>$${fmt(price)}</span></div>`;
  el.innerHTML=h;
}

function drawCurve(vk, sym, sd){
  const tag=sym.split('/')[0], svg=document.getElementById('cv_'+vk+'_'+tag);
  const c=sd.equity_curve||[]; if(!svg) return;
  if(c.length<2){svg.innerHTML='';return}
  const ys=c.map(p=>p[1]), start=sd.start_capital;
  let lo=Math.min(...ys,start),hi=Math.max(...ys,start);if(hi===lo){hi+=1;lo-=1}
  const W=600,H=42,pad=3,x=i=>i/(c.length-1)*W,yv=v=>pad+(1-(v-lo)/(hi-lo))*(H-2*pad);
  const up=ys[ys.length-1]>=start, col=up?'var(--grn)':'var(--red)';
  let dp='M'+x(0).toFixed(1)+' '+yv(ys[0]).toFixed(1);
  for(let i=1;i<c.length;i++) dp+=' L'+x(i).toFixed(1)+' '+yv(ys[i]).toFixed(1);
  svg.innerHTML=`<line x1="0" y1="${yv(start).toFixed(1)}" x2="${W}" y2="${yv(start).toFixed(1)}" stroke="var(--line)" stroke-dasharray="4 4"/><path d="${dp}" fill="none" stroke="${col}" stroke-width="1.5"/>`;
}

tick(); setInterval(tick, 3000);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(build_api()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Dashboard: http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
