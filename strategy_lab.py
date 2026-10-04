"""Editable BB lower-band strategy tests and earnings event hypothesis study."""
import numpy as np
import pandas as pd


def add_lab_indicators(prices, bb_period=20, bb_std=2.0, atr_period=14):
    p=prices.copy().sort_index()
    c=p['Close']; mid=c.rolling(bb_period).mean(); sd=c.rolling(bb_period).std(ddof=0)
    p['BB_Lower']=mid-bb_std*sd; p['BB_Middle']=mid; p['BB_Upper']=mid+bb_std*sd
    pc=c.shift(1)
    tr=pd.concat([p['High']-p['Low'],(p['High']-pc).abs(),(p['Low']-pc).abs()],axis=1).max(axis=1)
    p['ATR']=tr.ewm(alpha=1/atr_period,adjust=False,min_periods=atr_period).mean()
    p['Close_to_Lower_%']=(c/p['BB_Lower']-1)*100
    p['MA50']=c.rolling(50).mean();p['MA200']=c.rolling(200).mean()
    return p


def run_bb_strategy(prices, starting_cash=10000., entry_tolerance_pct=1., atr_stop=1.5,
                    target_r=2., max_hold=20, risk_pct=1., max_allocation_pct=100.,
                    commission_pct=.1, slippage_pct=.1, test_start=None, bb_period=20, bb_std=2., atr_period=14):
    """Signals at close, trades from next open; stop wins if stop/target both hit intraday."""
    if len(prices)<220:return {'error':'BB20・ATR・200日線を含む検証には、少なくとも220営業日程度の価格が必要です。'}
    if min(starting_cash,atr_stop,target_r,max_hold,risk_pct,max_allocation_pct)<=0 or entry_tolerance_pct<0:
        return {'error':'開始資金・Stop倍率・利確倍率・保有日数・損失率・投資上限を確認してください。'}
    p=add_lab_indicators(prices,bb_period,bb_std,atr_period)
    start_i = 0 if test_start is None else int(p.index.searchsorted(pd.Timestamp(test_start), side='left'))
    if start_i >= len(p)-1:return {'error':'選択した分析期間に売買用データがありません。'}
    cash=float(starting_cash); closed=[]; equity_rows=[]; active=None; skipped_overlap=0
    fees=max(0,float(commission_pct))/100; slip=max(0,float(slippage_pct))/100
    for i in range(start_i,len(p)):
        row=p.iloc[i]; day=p.index[i]
        closed_today=False
        if active is not None:
            held=i-active['entry_i']+1
            stop=active['stop']; target=active['target']
            exit_price=None;reason=None
            if row['Open']<=stop: exit_price=float(row['Open']);reason='Stop（寄付ギャップ）'
            elif row['Open']>=target: exit_price=target;reason='利確'
            elif row['Low']<=stop and row['High']>=target: exit_price=stop;reason='Stop（同日両方到達・保守計算）'
            elif row['Low']<=stop: exit_price=stop;reason='Stop'
            elif row['High']>=target: exit_price=target;reason='利確'
            elif held>=max_hold: exit_price=float(row['Close']);reason='最大保有日数'
            elif i==len(p)-1: exit_price=float(row['Close']);reason='期間末強制決済'
            if exit_price is not None:
                sell=exit_price*(1-slip);shares=active['shares'];buy=active['buy']
                exit_fee=shares*sell*fees
                pnl=shares*(sell-buy)-active['entry_fee']-exit_fee
                cash+=shares*sell-exit_fee
                closed.append({'Entry日':active['entry_day'],'Exit日':day,'保有営業日数':held,
                    'Entry価格':buy,'Exit価格':sell,'株数':shares,'Exit理由':reason,
                    '損益_通貨':pnl,'損益_%':pnl/(shares*buy+active['entry_fee'])*100,
                    '累積実現損益_通貨':cash-starting_cash})
                active=None
                closed_today=True
        if active is None and not closed_today and i>start_i and i<len(p)-1:
            sig=p.iloc[i-1]
            touched=(sig['Low']<=sig['BB_Lower']) if pd.notna(sig['BB_Lower']) else False
            near=(sig['Close_to_Lower_%']>=0 and sig['Close_to_Lower_%']<=entry_tolerance_pct) if pd.notna(sig['Close_to_Lower_%']) else False
            signal=bool(touched or near)
            if signal and pd.notna(sig['ATR']) and sig['ATR']>0:
                raw_entry=float(row['Open']);stop=raw_entry-float(sig['ATR'])*atr_stop
                risk_per_share=raw_entry-stop
                buy=raw_entry*(1+slip)
                equity=cash
                risk_budget=equity*risk_pct/100
                allocation=equity*max_allocation_pct/100
                shares=min(int(risk_budget/risk_per_share),int(allocation/buy))
                if shares>0:
                    entry_fee=shares*buy*fees
                    if shares*buy+entry_fee<=cash:
                        target=raw_entry+risk_per_share*target_r
                        active={'entry_i':i,'entry_day':day,'buy':buy,'raw_entry':raw_entry,'stop':stop,
                            'target':target,'shares':shares,'entry_fee':entry_fee}
                    else:skipped_overlap+=1
        marked=cash
        if active is not None:marked+=active['shares']*(float(row['Close'])-active['buy'])-active['entry_fee']
        if i>=start_i: equity_rows.append({'日付':day,'口座評価額_通貨':marked})
    trades=pd.DataFrame(closed);equity=pd.DataFrame(equity_rows).set_index('日付')
    peak=equity['口座評価額_通貨'].cummax().clip(lower=starting_cash)
    dd=(equity['口座評価額_通貨']/peak-1)*100
    final=float(equity.iloc[-1,0]); cumulative=(final/starting_cash-1)*100
    if trades.empty:
        summary={'取引数':0,'最終損益_通貨':final-starting_cash,'合計損益_%':cumulative,'勝ち取引数':0,
                 '負け取引数':0,'勝率_%':np.nan,'平均損益_通貨':np.nan,'最大資金減少_%':float(dd.min()),
                 '市場保有参考_%':(float(p.Close.iloc[-1])/float(p.Close.iloc[start_i])-1)*100,
                 '期間末現金':cash,'未決済ポジション':bool(active)}
    else:
        summary={'取引数':len(trades),'最終損益_通貨':final-starting_cash,'合計損益_%':cumulative,
            '勝ち取引数':int(trades['損益_通貨'].gt(0).sum()),'負け取引数':int(trades['損益_通貨'].lt(0).sum()),
            '勝率_%':float(trades['損益_通貨'].gt(0).mean()*100),'平均損益_通貨':float(trades['損益_通貨'].mean()),
            '最大資金減少_%':float(dd.min()),'市場保有参考_%':(float(p.Close.iloc[-1])/float(p.Close.iloc[start_i])-1)*100,
            '期間末現金':cash,'未決済ポジション':bool(active)}
    return {'error':None,'prices':p,'trades':trades,'equity':equity,'drawdown':dd,'summary':summary,
            'skipped_count':skipped_overlap}


def study_earnings(prices, event_dates, pre_days=20, rise_threshold_pct=5., pullback_threshold_pct=3.):
    p=add_lab_indicators(prices)
    if pre_days<2 or rise_threshold_pct<0 or pullback_threshold_pct<0:
        return {'error':'決算前の日数は2日以上、上昇・反落の幅は0以上を指定してください。'}
    events=pd.DatetimeIndex(pd.to_datetime(list(event_dates),errors='coerce')).dropna().normalize().unique().sort_values()
    rows=[]
    for event in events:
        # Input is the reported calendar date. Isolate only observations strictly before it.
        if len(p) and event>p.index[-1]:
            rows.append({'決算日':event,'状態':'株価期間の終了日より後','観察開始日':pd.NaT,'決算前最終日':p.index[-1],
                         '決算前上昇_%':np.nan,'上昇閾値到達':False,'上昇ピーク日':pd.NaT,'ピーク後反落_%':np.nan,'反落閾値到達':False})
            continue
        prior=p.loc[p.index<event]
        if len(prior)<pre_days:
            rows.append({'決算日':event,'状態':'決算前データ不足','観察開始日':pd.NaT,'決算前最終日':prior.index[-1] if len(prior) else pd.NaT,
                         '決算前上昇_%':np.nan,'上昇閾値到達':False,'上昇ピーク日':pd.NaT,'ピーク後反落_%':np.nan,'反落閾値到達':False})
            continue
        w=prior.tail(pre_days); base=float(w.Close.iloc[0]); peak_i=int(np.argmax(w.High.to_numpy())); peak=float(w.High.iloc[peak_i]); end=float(w.Close.iloc[-1])
        runup=(peak/base-1)*100; pullback=(end/peak-1)*100 if peak else np.nan
        rows.append({'決算日':event,'状態':'算出','観察開始日':w.index[0],'決算前最終日':w.index[-1],
            '観察開始終値':base,'期間最高値':peak,'決算前上昇_%':runup,
            '上昇閾値到達':bool(runup>=rise_threshold_pct),'上昇ピーク日':w.index[peak_i],
            '決算前最終終値':end,'ピーク後反落_%':pullback,
            '反落閾値到達':bool(pullback<=-pullback_threshold_pct)})
    table=pd.DataFrame(rows)
    valid=table.loc[table['状態'].eq('算出')] if not table.empty else table
    if valid.empty:summary={'登録決算数':len(events),'計算可能決算数':0,'上昇閾値到達数':0,'上昇閾値到達割合_%':np.nan,
        '利益確定売り仮説に沿う反落数':0,'上昇後反落割合_%':np.nan,'中央値上昇_%':np.nan,'中央値ピーク後反落_%':np.nan}
    else:summary={'登録決算数':len(events),'計算可能決算数':len(valid),
        '上昇閾値到達数':int(valid['上昇閾値到達'].sum()),'上昇閾値到達割合_%':float(valid['上昇閾値到達'].mean()*100),
        '利益確定売り仮説に沿う反落数':int((valid['上昇閾値到達']&valid['反落閾値到達']).sum()),
        '上昇後反落割合_%':float((valid['上昇閾値到達']&valid['反落閾値到達']).mean()*100),
        '中央値上昇_%':float(valid['決算前上昇_%'].median()),'中央値ピーク後反落_%':float(valid['ピーク後反落_%'].median())}
    return {'error':None,'summary':summary,'events':table,'pre_days':pre_days,
            'rise_threshold_pct':rise_threshold_pct,'pullback_threshold_pct':pullback_threshold_pct}
