"""
Trend-Filtered 5-Coin Basket -- PAPER TRADER (locked baseline: SMA200 / 4h / daily).
100% PAPER. Public Binance data only, no keys, no real orders. One run = one daily
decision, acting ONLY on fully-closed 4h bars. Maintains a JSON ledger + daily CSV log.

Rules (locked, see paper-trading brief):
  Universe: BTC ETH BNB XRP SOL /USDT spot. Long-only, no leverage.
  Eligible if latest CLOSED 4h close > SMA200(closed 4h closes).
  Target weight = 0.20 per eligible coin, remainder USDT cash.
  Costs: 0.10% fee + 0.02% slippage per leg. Cash yields 0%.
"""
import os, sys, json, csv, time, datetime as dt
import ccxt
HERE=os.path.dirname(os.path.abspath(__file__))
COINS=['BTC/USDT','ETH/USDT','BNB/USDT','XRP/USDT','SOL/USDT']
SMA_LEN=200; WEIGHT=0.20; FEE=0.001; SLIP=0.0002; TF='4h'; TF_MS=14400_000
LEDGER=os.path.join(HERE,'ledger.json'); DLOG=os.path.join(HERE,'daily_log.csv')
# pause/kill thresholds (from brief)
DD_REVIEW=40.0; DD_KILL=50.0

def now_ms(ex): return ex.milliseconds()
def closed_ohlcv(ex, sym, n=260, attempts=3):
    """Resilient fetch: up to `attempts` tries with short backoff (2s,4s,6s)
    to ride out transient ccxt RequestTimeout/NetworkError. Raises on
    persistent failure so the caller can skip THIS coin (not kill the run) --
    matches the hardening in the main adaptive bot. Strategy logic unchanged:
    still returns only fully-closed 4h bars (drops the forming bar)."""
    last=None
    for a in range(attempts):
        try:
            raw=ex.fetch_ohlcv(sym, TF, limit=n)
            nm=ex.milliseconds()
            return [r for r in raw if r[0]+TF_MS<=nm]
        except Exception as e:
            last=e; time.sleep(2*(a+1))
    raise last

def load_ledger(init_capital):
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"cash":init_capital,"units":{c:0.0 for c in COINS},"start_capital":init_capital,
            "peak_equity":init_capital,"realized_pnl":0.0,"n_trades":0,"created":dt.datetime.utcnow().isoformat()+"Z"}

def save_ledger(l): 
    tmp=LEDGER+'.tmp'; json.dump(l,open(tmp,'w'),indent=2); os.replace(tmp,LEDGER)

def main(init_capital=10000.0):
    ex=ccxt.binance({'enableRateLimit':True})
    price={}; sma={}; elig={}; skipped=[]
    for c in COINS:
        try:
            oh=closed_ohlcv(ex,c)
        except Exception:
            skipped.append(c); continue                      # persistent fetch failure -> skip THIS coin, keep run alive
        if len(oh)<SMA_LEN+1:
            skipped.append(c); continue                      # insufficient / gap -> skip coin
        closes=[r[4] for r in oh]
        price[c]=closes[-1]
        sma[c]=sum(closes[-SMA_LEN:])/SMA_LEN
        elig[c]=price[c]>sma[c]
    L=load_ledger(init_capital)
    # mark-to-market current equity
    inv=sum(L['units'][c]*price.get(c,0) for c in COINS)
    equity=L['cash']+inv
    if equity<=0: equity=init_capital
    # target weights
    target={c:(WEIGHT if elig.get(c,False) else 0.0) for c in COINS}
    # current weights
    curw={c:(L['units'][c]*price.get(c,0)/equity if equity else 0) for c in COINS}
    orders=[]; turnover=0.0; cost=0.0
    for c in COINS:
        if c in skipped: continue
        dw=target[c]-curw[c]
        if abs(dw)<1e-6: continue
        usd=dw*equity                                        # + buy, - sell
        legcost=abs(usd)*(FEE+SLIP)
        # execute against modeled fill
        fill=price[c]*(1+SLIP) if usd>0 else price[c]*(1-SLIP)
        units_delta=usd/fill
        L['units'][c]+=units_delta
        L['cash']-=usd; L['cash']-=legcost
        cost+=legcost; turnover+=abs(dw); L['n_trades']+=1
        orders.append({"coin":c,"side":"BUY" if usd>0 else "SELL","usd":round(usd,2),
                       "price":round(price[c],4),"cost":round(legcost,2)})
    # recompute equity post-trade
    inv=sum(L['units'][c]*price.get(c,0) for c in COINS); equity=L['cash']+inv
    L['peak_equity']=max(L['peak_equity'],equity)
    dd=100*(L['peak_equity']-equity)/L['peak_equity'] if L['peak_equity'] else 0
    save_ledger(L)
    # daily log row
    ts=dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    eligset=[c for c in COINS if elig.get(c)]
    exp=100*inv/equity if equity else 0
    newf=not os.path.exists(DLOG)
    with open(DLOG,'a',newline='') as f:
        w=csv.writer(f)
        if newf: w.writerow(['ts','equity','cash','exposure_pct','eligible','n_eligible','dd_from_peak_pct','turnover','cost','orders','skipped'])
        w.writerow([ts,round(equity,2),round(L['cash'],2),round(exp,1),'|'.join(s.split('/')[0] for s in eligset),
                    len(eligset),round(dd,2),round(turnover,3),round(cost,2),len(orders),'|'.join(skipped)])
    # pause/kill check
    alert='OK'
    if dd>DD_KILL: alert='KILL: drawdown %.1f%% > %.0f%% -> go to cash & review'%(dd,DD_KILL)
    elif dd>DD_REVIEW: alert='REVIEW: drawdown %.1f%% > %.0f%%'%(dd,DD_REVIEW)
    return dict(ts=ts,equity=equity,cash=L['cash'],exposure=exp,eligible=eligset,
                target=target,curw=curw,orders=orders,turnover=turnover,cost=cost,
                dd=dd,skipped=skipped,price=price,sma=sma,alert=alert)

if __name__=='__main__':
    cap=float(sys.argv[1]) if len(sys.argv)>1 else 10000.0
    r=main(cap)
    print('=== TREND-BASKET PAPER DECISION  %s ==='%r['ts'])
    print('equity $%.2f  cash $%.2f  exposure %.1f%%  drawdown-from-peak %.2f%%'%(r['equity'],r['cash'],r['exposure'],r['dd']))
    print('signal (closed 4h close vs SMA200):')
    for c in COINS:
        p=r['price'].get(c); s=r['sma'].get(c)
        if p is None: print('  %-9s  SKIPPED (insufficient/gap)'%c); continue
        print('  %-9s close=%-11.4f SMA200=%-11.4f -> %s'%(c,p,s,'ELIGIBLE (long 20%)' if p>s else 'cash'))
    print('eligible set: %s (%d/5), target invested %.0f%%'%(', '.join(x.split('/')[0] for x in r['eligible']) or 'NONE', len(r['eligible']), 20*len(r['eligible'])))
    print('orders generated this cycle:')
    if not r['orders']: print('  (none)')
    for o in r['orders']: print('  %-4s %-9s $%9.2f @ %.4f  cost $%.2f'%(o['side'],o['coin'],o['usd'],o['price'],o['cost']))
    print('turnover %.3f  cost $%.2f'%(r['turnover'],r['cost']))
    print('ALERT:', r['alert'])
    print('ledger: %s   daily log: %s'%(LEDGER,DLOG))
