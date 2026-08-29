"""
READ-ONLY live dashboard for the adaptive Binance Spot paper trader.
Stdlib only, same architecture as the legacy dashboard.py (BaseHTTPRequestHandler
+ ThreadingHTTPServer). Never imports or calls into adaptive/runner.py or
adaptive/research_service.py, never writes to any file under adaptive_runtime/
-- it only opens adaptive_runtime/*.json, *.jsonl, *.csv for reading, plus a
harmless `launchctl list` shell-out (read-only process inspection) to report
each service's live RUNNING/STOPPED status.

    python3 adaptive/dashboard.py --port 8788   # http://localhost:8788

Fields the running trading process doesn't currently persist (e.g. a
per-timeframe trend/regime breakdown, or a full opportunity-score
breakdown per candidate) are shown as "not available" rather than
invented -- see the module docstring in adaptive/runner.py for what IS
written to dashboard.json/decisions.jsonl today.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNTIME_DIR = os.path.join(ROOT, "adaptive_runtime")

SERVICE_LABELS = {
    "adaptive": "com.btcpaper.adaptive",
    "research": "com.btcpaper.adaptive.research",
    "legacy_bot": "com.btcpaper.bot",
    "shocksol": "com.btcpaper.shocksol",
}


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl_tail(path, n=40):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            lines = f.readlines()
    except Exception:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out[::-1]


def _read_csv_tail(path, n=25):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    return rows[-n:][::-1]


def _service_status(label):
    """Exact match on launchctl's tab-separated Label column -- a substring/
    endswith check would incorrectly match "com.btcpaper.adaptive" against
    the "com.btcpaper.adaptive.research" line (which contains it as a
    prefix), conflating the two services' PIDs."""
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[-1] == label:
                return ("RUNNING", parts[0]) if parts[0] != "-" else ("STOPPED", None)
        return "STOPPED", None
    except Exception:
        return "UNKNOWN", None


def _risk_limits():
    """Reads the static default RiskLimits() dataclass -- a config
    definition, not live state; importing it does not touch or restart
    the trading engine."""
    sys.path.insert(0, ROOT)
    try:
        from adaptive.risk_engine import RiskLimits
        r = RiskLimits()
        return {
            "risk_per_trade_pct": r.risk_per_trade_pct,
            "max_portfolio_heat_pct": r.max_portfolio_heat_pct,
            "max_symbol_allocation_pct": r.max_symbol_allocation_pct,
            "max_total_crypto_allocation_pct": r.max_total_crypto_allocation_pct,
            "daily_loss_limit_pct": r.daily_loss_limit_pct,
            "max_drawdown_limit_pct": r.max_drawdown_limit_pct,
            "consecutive_loss_breaker": r.consecutive_loss_breaker,
            "min_cash_reserve_pct": r.min_cash_reserve_pct,
        }
    except Exception:
        return {}


def build_api():
    dash = _read_json(os.path.join(RUNTIME_DIR, "dashboard.json")) or {}
    research_status = _read_json(os.path.join(RUNTIME_DIR, "research_status.json")) or {}
    decisions = _read_jsonl_tail(os.path.join(RUNTIME_DIR, "decisions.jsonl"), 40)
    adaptation_log = _read_jsonl_tail(os.path.join(RUNTIME_DIR, "adaptation_log.jsonl"), 40)
    trades = _read_csv_tail(os.path.join(RUNTIME_DIR, "trades.csv"), 25)

    adaptive_status, adaptive_pid = _service_status(SERVICE_LABELS["adaptive"])
    research_status_svc, research_pid = _service_status(SERVICE_LABELS["research"])
    bot_status, bot_pid = _service_status(SERVICE_LABELS["legacy_bot"])
    shocksol_status, shocksol_pid = _service_status(SERVICE_LABELS["shocksol"])

    # most recent cycle_ranking decision -- the freshest opportunity snapshot
    last_ranking = next((d for d in decisions if d.get("action") == "cycle_ranking"), None)
    desired_positions = len(last_ranking.get("selected", [])) if last_ranking else None

    return {
        "generated_at_iso": dash.get("saved_at_iso") or dash.get("written_at_iso"),
        "services": {
            "adaptive": {"status": adaptive_status, "pid": adaptive_pid},
            "research": {"status": research_status_svc, "pid": research_pid},
            "legacy_bot": {"status": bot_status, "pid": bot_pid},
            "shocksol": {"status": shocksol_status, "pid": shocksol_pid},
        },
        "portfolio": dash,  # equity/cash/return_pct/drawdown_pct/daily_pnl_*/portfolio_heat_*/positions/market/ai_brain
        "research": research_status,
        "decisions": decisions,
        "last_ranking": last_ranking,
        "desired_positions": desired_positions,
        "adaptation_log": adaptation_log,
        "trades": trades,
        "risk_limits": _risk_limits(),
        "data_sources": [
            "adaptive_runtime/dashboard.json", "adaptive_runtime/research_status.json",
            "adaptive_runtime/decisions.jsonl", "adaptive_runtime/adaptation_log.jsonl",
            "adaptive_runtime/trades.csv",
        ],
    }


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Adaptive Spot AI Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg:#0b0f14; --panel:#121821; --panel2:#161d29; --border:#232c3a;
  --text:#e6edf3; --dim:#8b98a9; --green:#3fb950; --red:#f85149; --yellow:#d29922;
  --blue:#58a6ff; --accent:#7ee787;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       margin:0; padding:20px; font-size:14px; }
h1 { font-size:20px; margin:0 0 4px; }
h2 { font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); margin:28px 0 10px; border-bottom:1px solid var(--border); padding-bottom:6px; }
.banner { background:linear-gradient(90deg,#0d1b0f,#0b0f14); border:1px solid #1f3a24; color:var(--accent);
          padding:10px 16px; border-radius:8px; font-weight:600; letter-spacing:.03em; margin-bottom:16px; }
.top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }
.pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.pill.run { background:#0d2818; color:var(--green); border:1px solid #1f4a2c; }
.pill.stop { background:#2a1215; color:var(--red); border:1px solid #4a1f22; }
.pill.unknown { background:#2a2412; color:var(--yellow); border:1px solid #4a3f1f; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.grid3 { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.stat-label { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
.stat-value { font-size:20px; font-weight:600; margin-top:2px; }
.pos { color:var(--green); } .neg { color:var(--red); } .neu { color:var(--text); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); white-space:nowrap; }
th { color:var(--dim); font-weight:500; font-size:11px; text-transform:uppercase; }
tr:hover td { background:#0f151f; }
.badge { padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600; }
.badge.buy { background:#0d2818; color:var(--green); }
.badge.hold { background:#1a2230; color:var(--blue); }
.badge.skip { background:#20232a; color:var(--dim); }
.badge.sell { background:#2a1215; color:var(--red); }
.badge.promoted { background:#0d2818; color:var(--green); }
.badge.rejected { background:#20232a; color:var(--dim); }
.badge.rollback { background:#2a1215; color:var(--red); }
.muted { color:var(--dim); font-style:italic; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.subgrid { display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; margin-top:8px; }
.subgrid .k { color:var(--dim); font-size:12px; } .subgrid .v { font-size:12px; text-align:right; }
.flatnote { color:var(--dim); padding:14px; text-align:center; border:1px dashed var(--border); border-radius:8px; }
.footer { color:var(--dim); font-size:11px; margin-top:30px; text-align:center; }
.refresh-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); margin-right:6px; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.3;} }
.na { color:var(--dim); }
</style>
</head>
<body>
<div class="banner">BINANCE SPOT ONLY &nbsp;·&nbsp; PAPER TRADING &nbsp;·&nbsp; NO FUTURES &nbsp;·&nbsp; NO MARGIN &nbsp;·&nbsp; NO REAL ORDERS</div>

<div class="top">
  <div>
    <h1>Adaptive Binance Spot AI Paper Trader</h1>
    <div class="muted" id="genat">loading…</div>
  </div>
  <div id="svc-pills"></div>
</div>

<h2>Portfolio</h2>
<div class="grid" id="portfolio-cards"></div>

<h2>Live Market</h2>
<div class="grid3" id="market-cards"></div>

<h2>Positions</h2>
<div class="card"><div id="positions"></div></div>

<h2>AI Brain</h2>
<div class="card" id="ai-brain"></div>

<h2>Strategy Brain</h2>
<div class="card" id="strategy-brain"></div>

<h2>Autonomous Research</h2>
<div class="card" id="research"></div>

<h2>Risk</h2>
<div class="card" id="risk"></div>

<h2>Trade History</h2>
<div class="card"><div id="trades"></div></div>

<h2>Decision Log</h2>
<div class="card"><div id="decisions"></div></div>

<h2>Adaptation History</h2>
<div class="card"><div id="adaptation"></div></div>

<div class="footer"><span class="refresh-dot"></span>auto-refreshing every 7s — read-only, adaptive_runtime/* files only</div>

<script>
const fmt = (v, d=2) => (v===null||v===undefined||Number.isNaN(v)) ? '<span class="na">—</span>' : Number(v).toFixed(d);
const pct = (v, d=2) => (v===null||v===undefined) ? '<span class="na">—</span>' : Number(v).toFixed(d)+'%';
const cls = v => v>0?'pos':(v<0?'neg':'neu');
const esc = s => (s===null||s===undefined) ? '' : String(s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ago = iso => { if(!iso) return '—'; const d=(Date.now()-new Date(iso).getTime())/1000; if(d<60) return Math.floor(d)+'s ago'; if(d<3600) return Math.floor(d/60)+'m ago'; return Math.floor(d/3600)+'h ago'; };

function pill(status, pid){
  const cls = status==='RUNNING'?'run':(status==='STOPPED'?'stop':'unknown');
  return `<span class="pill ${cls}">${status}${pid?(' · pid '+pid):''}</span>`;
}

async function refresh(){
  let d;
  try { d = await (await fetch('/api/state')).json(); } catch(e) { return; }

  document.getElementById('genat').textContent = 'last updated ' + (d.generated_at_iso || '—') + ' (' + ago(d.generated_at_iso) + ')';
  document.getElementById('svc-pills').innerHTML =
    'adaptive ' + pill(d.services.adaptive.status, d.services.adaptive.pid) + '&nbsp;&nbsp;' +
    'research ' + pill(d.services.research.status, d.services.research.pid);

  const p = d.portfolio || {};
  document.getElementById('portfolio-cards').innerHTML = `
    <div class="card"><div class="stat-label">Total Equity</div><div class="stat-value">$${fmt(p.equity)}</div></div>
    <div class="card"><div class="stat-label">Available USDT Cash</div><div class="stat-value">$${fmt(p.cash)}</div></div>
    <div class="card"><div class="stat-label">Total Return</div><div class="stat-value ${cls(p.return_pct)}">${pct(p.return_pct)}</div></div>
    <div class="card"><div class="stat-label">Daily PnL</div><div class="stat-value ${cls(p.daily_pnl_usdt)}">$${fmt(p.daily_pnl_usdt)} (${pct(p.daily_pnl_pct)})</div></div>
    <div class="card"><div class="stat-label">Drawdown</div><div class="stat-value ${p.drawdown_pct>0?'neg':'neu'}">${pct(p.drawdown_pct)}</div></div>
    <div class="card"><div class="stat-label">Portfolio Heat</div><div class="stat-value">${pct(p.portfolio_heat_pct)}</div></div>
    <div class="card"><div class="stat-label">Crypto Allocation</div><div class="stat-value">${pct(p.crypto_allocation_pct)}</div></div>
    <div class="card"><div class="stat-label">Risk State</div><div class="stat-value">${esc(p.risk_state||'—')}</div></div>
  `;

  const market = (p.market)||{};
  const syms = ['BTC/USDT','ETH/USDT','SOL/USDT'];
  document.getElementById('market-cards').innerHTML = syms.map(s=>{
    const m = market[s]||{};
    const spread = m.spread_pct;
    return `<div class="card">
      <div class="stat-label">${s}</div>
      <div class="stat-value">$${fmt(m.last_price, m.last_price>100?2:4)}</div>
      <div class="subgrid">
        <div class="k">Bid</div><div class="v">${fmt(m.bid,4)}</div>
        <div class="k">Ask</div><div class="v">${fmt(m.ask,4)}</div>
        <div class="k">Spread</div><div class="v">${spread!==undefined?pct(spread,4):'<span class=na>—</span>'}</div>
        <div class="k">5m Trend</div><div class="v na">not persisted</div>
        <div class="k">15m Trend</div><div class="v na">not persisted</div>
        <div class="k">1h Regime</div><div class="v na">not persisted</div>
        <div class="k">4h Context</div><div class="v na">not persisted</div>
        <div class="k">Data Freshness</div><div class="v">${m.candle_age_sec!==undefined?Math.round(m.candle_age_sec)+'s':'<span class=na>—</span>'}</div>
        <div class="k">Last Candle</div><div class="v">${m.last_closed_bar_ts?new Date(m.last_closed_bar_ts).toISOString().slice(11,19)+'Z':'<span class=na>—</span>'}</div>
        <div class="k">Gap Flag</div><div class="v">${m.gap_flag?'<span class=neg>YES</span>':'no'}</div>
      </div>
    </div>`;
  }).join('');

  const positions = (p.positions)||{};
  const posKeys = Object.keys(positions);
  if(posKeys.length===0){
    document.getElementById('positions').innerHTML = '<div class="flatnote">NO OPEN POSITION</div>';
  } else {
    document.getElementById('positions').innerHTML = `<table><thead><tr>
      <th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Current</th><th>Value</th>
      <th>Unrealized PnL</th><th>PnL %</th><th>Stop</th><th>Strategy</th><th>Regime</th><th>Confidence</th>
      </tr></thead><tbody>` + posKeys.map(sym=>{
        const pos = positions[sym];
        const val = pos.qty*(pos.current_price||pos.entry_price);
        const pnlPct = pos.entry_price ? (pos.unrealized_pnl/(pos.qty*pos.entry_price)*100) : null;
        return `<tr>
          <td>${esc(sym)}</td><td>LONG</td><td>${fmt(pos.qty,6)}</td><td>$${fmt(pos.entry_price,4)}</td>
          <td>$${fmt(pos.current_price,4)}</td><td>$${fmt(val)}</td>
          <td class="${cls(pos.unrealized_pnl)}">$${fmt(pos.unrealized_pnl)}</td>
          <td class="${cls(pnlPct)}">${pct(pnlPct)}</td>
          <td>$${fmt(pos.stop_price,4)}</td><td>${esc(pos.strategy)}</td><td>${esc(pos.regime)}</td><td>${pct((pos.confidence||0)*100,0)}</td>
        </tr>`;
      }).join('') + '</tbody></table>';
  }

  const ai = (p.ai_brain)||{};
  const champs = ai.champions||{};
  const desired = d.desired_positions;
  let desiredReason = 'no recent ranking cycle available';
  if(d.last_ranking){
    const n = (d.last_ranking.selected||[]).length;
    desiredReason = n===0 ? 'no market currently exceeds the minimum quality threshold' : (d.last_ranking.selected.join(', ') + ' cleared ranking this cycle');
  }
  document.getElementById('ai-brain').innerHTML = `
    <div class="subgrid" style="grid-template-columns:1fr 2fr">
      ${Object.keys(champs).map(s=>`<div class="k">${esc(s)}</div><div class="v mono">v${champs[s].version} (${esc(champs[s].status)}) · ${esc(champs[s].hash)} · ${esc(JSON.stringify(champs[s].parameters))}</div>`).join('')}
      <div class="k">Risk State</div><div class="v">${esc(ai.risk_state||'—')}</div>
    </div>
    <div style="margin-top:14px; font-weight:600;">AI CURRENTLY WANTS: ${desired===null||desired===undefined?'<span class=na>—</span>':desired} / 3 POSITIONS</div>
    <div class="muted">"${desired===0?'0 positions — '+desiredReason:desiredReason}"</div>
    <div style="margin-top:14px;" class="muted">Per-candidate score breakdown (expected edge, regime fit, robustness, friction/correlation penalty) is not
    currently written to decisions.jsonl by the running process -- only symbol/strategy/score are logged per cycle-ranking event.
    Latest candidates: ${d.last_ranking && d.last_ranking.candidates && d.last_ranking.candidates.length ? d.last_ranking.candidates.map(c=>esc(c.symbol)+'/'+esc(c.strategy)+' score='+fmt(c.score,3)).join(', ') : '<span class=na>none this cycle</span>'}</div>
  `;

  // Strategy brain -- inferred only from the latest cycle's candidate list (real data), never fabricated BULLISH/BEARISH
  const activeStrats = new Set((d.last_ranking && d.last_ranking.candidates || []).map(c=>c.symbol+'|'+c.strategy));
  const strategies = ['trend_momentum','volatility_breakout','mean_reversion','shock_continuation'];
  document.getElementById('strategy-brain').innerHTML = `<table><thead><tr><th>Strategy</th>${syms.map(s=>`<th>${s}</th>`).join('')}</tr></thead><tbody>` +
    strategies.map(st => `<tr><td>${st}</td>` + syms.map(s => {
      const active = activeStrats.has(s+'|'+st);
      return `<td>${active?'<span class="badge buy">ACTIVE CANDIDATE</span>':'<span class="muted">no signal this cycle</span>'}</td>`;
    }).join('') + '</tr>').join('') + '</tbody></table>' +
    '<div class="muted" style="margin-top:8px;">Full BULLISH/BEARISH/NEUTRAL/INACTIVE state per strategy is not currently persisted -- shown here is only whether each (symbol, strategy) appeared as an active LONG candidate in the most recent decision cycle.</div>';

  const r = d.research || {};
  const lr = r.last_result || {};
  document.getElementById('research').innerHTML = `
    <div class="subgrid" style="grid-template-columns:1fr 2fr">
      <div class="k">Research Worker</div><div class="v">${pill(d.services.research.status, d.services.research.pid)}</div>
      <div class="k">Last Research Run</div><div class="v">${esc(r.last_full_run_iso||'—')}</div>
      <div class="k">Next Research Run</div><div class="v">${esc(r.next_full_run_iso||'—')}</div>
      <div class="k">Challengers Evaluated (last cycle)</div><div class="v">${lr.n_evaluated!==undefined?lr.n_evaluated:'<span class=na>—</span>'}</div>
      <div class="k">Challengers Promoted (last cycle)</div><div class="v">${lr.n_promoted!==undefined?lr.n_promoted:'<span class=na>—</span>'}</div>
      <div class="k">Challengers Rejected (last cycle)</div><div class="v">${lr.n_evaluated!==undefined?(lr.n_evaluated-(lr.n_promoted||0)):'<span class=na>—</span>'}</div>
    </div>
    <div style="margin-top:12px; font-weight:600;">${(lr.n_promoted||0)>0 ? (lr.n_promoted+' challenger(s) promoted this cycle') : 'NO CHALLENGER PASSED — CHAMPION UNCHANGED'}</div>
    <div class="muted">This is a valid state.</div>
  `;

  const rl = d.risk_limits||{};
  document.getElementById('risk').innerHTML = `<div class="subgrid" style="grid-template-columns:1fr 1fr">
    <div class="k">Portfolio Heat / Limit</div><div class="v">${pct(p.portfolio_heat_pct)} / ${pct(rl.max_portfolio_heat_pct)}</div>
    <div class="k">Daily Loss / Limit</div><div class="v">${p.daily_pnl_pct<0?pct(-p.daily_pnl_pct):'0.00%'} / ${pct(rl.daily_loss_limit_pct)}</div>
    <div class="k">Drawdown / Limit</div><div class="v">${pct(p.drawdown_pct)} / ${pct(rl.max_drawdown_limit_pct)}</div>
    <div class="k">Consecutive Losses / Limit</div><div class="v">${p.consecutive_losses!==undefined?p.consecutive_losses:'<span class=na>—</span>'} / ${rl.consecutive_loss_breaker!==undefined?rl.consecutive_loss_breaker:'<span class=na>—</span>'}</div>
    <div class="k">Risk Per Trade / Limit</div><div class="v">${pct(rl.risk_per_trade_pct)} (config)</div>
    <div class="k">Max Symbol Allocation</div><div class="v">${pct(rl.max_symbol_allocation_pct)}</div>
    <div class="k">Crypto Allocation</div><div class="v">${pct(p.crypto_allocation_pct)} / ${pct(rl.max_total_crypto_allocation_pct)}</div>
    <div class="k">Cash Reserve</div><div class="v">${pct(p.cash_allocation_pct)} (min ${pct(rl.min_cash_reserve_pct)})</div>
  </div>`;

  const trades = d.trades||[];
  document.getElementById('trades').innerHTML = trades.length===0
    ? '<div class="flatnote">No completed trades yet</div>'
    : `<table><thead><tr><th>Exit Time</th><th>Symbol</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>Net PnL</th><th>Fees</th><th>Exit Reason</th></tr></thead><tbody>` +
      trades.map(t=>`<tr><td>${esc(t.exit_ts_ms)}</td><td>${esc(t.symbol)}</td><td>${esc(t.strategy)}</td>
        <td>$${fmt(t.entry_price,4)}</td><td>$${fmt(t.exit_price,4)}</td>
        <td class="${cls(parseFloat(t.net_pnl))}">$${fmt(t.net_pnl)}</td>
        <td>$${fmt((parseFloat(t.entry_fee)||0)+(parseFloat(t.exit_fee)||0))}</td><td>${esc(t.exit_reason)}</td></tr>`).join('') + '</tbody></table>';

  const decisions = (d.decisions||[]).filter(x=>x.action!=='cycle_ranking').slice(0,25);
  document.getElementById('decisions').innerHTML = decisions.length===0
    ? '<div class="flatnote">No recent per-symbol decisions (only cycle-ranking snapshots so far)</div>'
    : `<table><thead><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Detail</th></tr></thead><tbody>` +
      decisions.map(x=>`<tr><td>${esc(x.ts_iso)}</td><td>${esc(x.symbol)}</td><td>${esc(x.action)}</td>
        <td class="mono">${esc(JSON.stringify(Object.fromEntries(Object.entries(x).filter(([k])=>!['ts_iso','symbol','action'].includes(k)))))}</td></tr>`).join('') + '</tbody></table>';

  const adapt = d.adaptation_log||[];
  document.getElementById('adaptation').innerHTML = adapt.length===0
    ? '<div class="flatnote">No adaptation events yet</div>'
    : `<table><thead><tr><th>Time</th><th>Strategy</th><th>Result</th><th>Detail</th></tr></thead><tbody>` +
      adapt.map(x=>{
        const badge = x.action==='promoted'?'promoted':(x.action==='rollback'?'rollback':'rejected');
        const detail = x.action==='promoted'
          ? `${esc(JSON.stringify(x.old_params))} → ${esc(JSON.stringify(x.new_params))} (${esc(x.reason||'')})`
          : (x.action==='rollback' ? esc(x.reason||'') : esc(JSON.stringify(x.reasons||{})));
        return `<tr><td>${esc(x.ts_iso)}</td><td>${esc(x.strategy)}</td><td><span class="badge ${badge}">${esc(x.action).toUpperCase()}</span></td><td class="mono">${detail}</td></tr>`;
      }).join('') + '</tbody></table>';
}

refresh();
setInterval(refresh, 7000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- avoid spamming launchd's stdout log for every poll

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(build_api(), default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path.startswith("/?"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"adaptive dashboard on http://localhost:{args.port} (read-only, {RUNTIME_DIR})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
