from __future__ import annotations
import math
import numpy as np
import pandas as pd
import yfinance as yf

APP_VERSION="2.0.0"
BB_PERIOD=20; BB_STD=2.0; BW_LOOKBACK=125; CASE_WINDOW_DAYS=3; HORIZONS=(5,10,20)

def _clean_ticker(t): return str(t or "").strip().upper()

def download_prices(ticker,start_date,end_date):
    ticker=_clean_ticker(ticker); s=pd.Timestamp(start_date).normalize(); e=pd.Timestamp(end_date).normalize()
    if not ticker or e<s: return pd.DataFrame()
    try:
        df=yf.download(ticker,start=s.strftime("%Y-%m-%d"),end=(e+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),interval="1d",auto_adjust=False,progress=False,multi_level_index=False)
    except Exception: return pd.DataFrame()
    if df is None or df.empty: return pd.DataFrame()
    df=df.copy()
    if getattr(df.index,"tz",None) is not None: df.index=df.index.tz_localize(None)
    df.index=pd.to_datetime(df.index).normalize()
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns: return pd.DataFrame()
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["Open","High","Low","Close"]).sort_index()

def add_indicators(df,atr_period=14):
    df=df.copy(); df["BB_Middle"]=df.Close.rolling(20).mean(); sd=df.Close.rolling(20).std(ddof=0)
    df["BB_Upper"]=df.BB_Middle+2*sd; df["BB_Lower"]=df.BB_Middle-2*sd
    df["BandWidth"]=(df.BB_Upper-df.BB_Lower)/df.BB_Middle
    mn=df.BandWidth.rolling(125).min(); mx=df.BandWidth.rolling(125).max(); den=mx-mn
    df["Normalized_BandWidth"]=np.where(den.ne(0),(df.BandWidth-mn)/den,np.nan); df["Low_BandWidth_Zone"]=df.Normalized_BandWidth.le(.20)
    df["MA50"]=df.Close.rolling(50).mean(); df["MA50_Deviation_Pct"]=(df.Close/df.MA50-1)*100
    df["Return_20D_Pct"]=df.Close.pct_change(20)*100; ret=df.Close.pct_change(); df["Vol_20D_Annualized_Pct"]=ret.rolling(20).std(ddof=0)*np.sqrt(252)*100
    pc=df.Close.shift(1); tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    df["ATR"]=tr.ewm(alpha=1/atr_period,adjust=False,min_periods=atr_period).mean(); df["ATR_Pct"]=df.ATR/df.Close*100
    df["Close_to_Lower_Pct"]=(df.Close/df.BB_Lower-1)*100; df["Low_to_Lower_Pct"]=(df.Low/df.BB_Lower-1)*100
    df["Prev_Low"]=df.Low.shift(1); df["Prev_Close"]=df.Close.shift(1); df["Prev_High"]=df.High.shift(1)
    df["Higher_Low"]=df.Low>df.Prev_Low; df["Close_Up"]=df.Close>df.Prev_Close
    df["Decline_Stop"]=df.Higher_Low & df.Close_Up; df["Rebound_Start"]=df.Close>df.Prev_High
    return df

def resolve_case_date(df,d):
    d=pd.Timestamp(d).normalize(); x=df.index[df.index>=d]
    if not len(x): return None,"入力日以降の営業日データなし"
    a=x[0]; return (a,"入力日を使用") if a==d else (a,f"入力日は非取引日のため次営業日 {a.date()} を使用")

def case_summary(df,d,atrp):
    r=df.loc[d]
    state="終値がBB下限以下" if r.Close<=r.BB_Lower else ("日中安値がBB下限到達・終値は上" if r.Low<=r.BB_Lower else "BB下限未到達（距離を確認）")
    return pd.DataFrame([{"ケース日":d.date(),"Open":r.Open,"High":r.High,"Low":r.Low,"Close":r.Close,"BB下限":r.BB_Lower,"BB中央":r.BB_Middle,"BB上限":r.BB_Upper,"終値-BB下限距離_%":r.Close_to_Lower_Pct,"安値-BB下限距離_%":r.Low_to_Lower_Pct,"BB状態":state,"BandWidth":r.BandWidth,"正規化BandWidth":r.Normalized_BandWidth,"低BandWidth帯":bool(r.Low_BandWidth_Zone),f"ATR{atrp}":r.ATR,f"ATR{atrp}_%":r.ATR_Pct,"過去20日年率Vol_%":r.Vol_20D_Annualized_Pct,"20日騰落率_%":r.Return_20D_Pct,"MA50乖離_%":r.MA50_Deviation_Pct}])

def pre_summary(df,start,d,atrp):
    p=df.loc[(df.index>=start)&(df.index<=d)]
    if p.empty:return pd.DataFrame()
    a=p.iloc[0]; z=p.iloc[-1]
    return pd.DataFrame([{"分析開始営業日":p.index[0].date(),"Day0":d.date(),"営業日数":len(p),"開始Close":a.Close,"Day0_Close":z.Close,"期間騰落率_%":(z.Close/a.Close-1)*100,"期間最高Close":p.Close.max(),"期間最低Close":p.Close.min(),f"Day0_ATR{atrp}":z.ATR,f"Day0_ATR{atrp}_%":z.ATR_Pct,"Day0_20日騰落率_%":z.Return_20D_Pct,"Day0_MA50乖離_%":z.MA50_Deviation_Pct}])

def find_signals(df,d):
    p0=df.index.get_loc(d); rows=[]
    for p in range(p0,min(len(df),p0+4)):
        r=df.iloc[p]; rows.append({"Day":p-p0,"日付":df.index[p].date(),"安値":r.Low,"終値":r.Close,"前日安値":r.Prev_Low,"前日終値":r.Prev_Close,"前日高値":r.Prev_High,"Higher_Low":bool(r.Higher_Low),"Close_Up":bool(r.Close_Up),"下落停止":bool(r.Decline_Stop),"反発開始":bool(r.Rebound_Start)})
    return pd.DataFrame(rows)

def first_signal(df,d,col):
    p0=df.index.get_loc(d)
    for p in range(p0,min(len(df),p0+4)):
        if bool(df.iloc[p][col]): return p
    return None

def build_risk_design(df,d,mults):
    rows=[]; p0=df.index.get_loc(d)
    for col,label in [("Decline_Stop","下落停止"),("Rebound_Start","反発開始")]:
        p=first_signal(df,d,col)
        if p is None or p+1>=len(df): rows.append({"シグナル":label,"Stop方式":"-","状態":"Day0～Day3に成立なし、または翌営業日データなし"}); continue
        entry=float(df.iloc[p+1].Open); atr=float(df.iloc[p].ATR); structure=float(df.iloc[p0:p+1].Low.min())
        methods=[("価格構造",structure,np.nan)]+[(f"ATR×{float(m):g}",entry-atr*float(m),float(m)) for m in mults if pd.notna(atr) and atr>0]
        for name,stop,m in methods:
            risk=entry-stop
            if risk<=0: continue
            rows.append({"シグナル":label,"シグナル日":df.index[p].date(),"Entry日":df.index[p+1].date(),"Entry_Open":entry,"Stop方式":name,"ATR倍率":m,"ATR_シグナル日":atr,"Stop":stop,"1R":risk,"1R_%":risk/entry*100,"+1R":entry+risk,"+1.5R":entry+1.5*risk,"+2R":entry+2*risk,"状態":"R計算可能"})
    return pd.DataFrame(rows)

def _hit(df,ep,stop,target,last):
    entry=float(df.iloc[ep].Open); risk=entry-stop; avail=min(last,len(df)-1); highs=[]; lows=[]
    for p in range(ep,avail+1):
        r=df.iloc[p]; o,h,l=float(r.Open),float(r.High),float(r.Low); highs.append(h); lows.append(l); dt=df.index[p]
        if o<=stop:return ["Stop先着",dt,o,"StopギャップOpen決済" if o<stop else "Stop決済",(o-entry)/risk,(max(highs)-entry)/risk,(entry-min(lows))/risk,np.nan]
        if o>=target:return ["Target先着",dt,o,"TargetギャップOpen決済" if o>target else "Target決済",(o-entry)/risk,(max(highs)-entry)/risk,(entry-min(lows))/risk,np.nan]
        hs=l<=stop; ht=h>=target
        if hs and ht:return ["同日両方到達・順序不明",dt,np.nan,"順序不明",np.nan,(max(highs)-entry)/risk,(entry-min(lows))/risk,np.nan]
        if hs:return ["Stop先着",dt,stop,"Stop決済",-1.,(max(highs)-entry)/risk,(entry-min(lows))/risk,np.nan]
        if ht:return ["Target先着",dt,target,"Target決済",2.,(max(highs)-entry)/risk,(entry-min(lows))/risk,np.nan]
    if avail<last:return ["将来データ不足・未決着",pd.NaT,np.nan,"データ不足",np.nan,(max(highs)-entry)/risk if highs else np.nan,(entry-min(lows))/risk if lows else np.nan,np.nan]
    close=float(df.iloc[last].Close); rr=(close-entry)/risk
    return ["期間内未到達",df.index[last],close,"期間末終値決済",rr,(max(highs)-entry)/risk,(entry-min(lows))/risk,rr]

def net_r(entry,exitp,risk,comm,slip):
    buy=entry*(1+slip); sell=exitp*(1-slip); return (sell-buy-buy*comm-sell*comm)/risk

def build_outcomes(df,design,end,comm,slip):
    rows=[]
    if design.empty:return pd.DataFrame()
    endpos=df.index.get_loc(df.index[df.index<=end][-1])
    for _,d in design[design.状態=="R計算可能"].iterrows():
        ep=df.index.get_loc(pd.Timestamp(d.Entry日)); entry=float(d.Entry_Open); stop=float(d.Stop); risk=float(d["1R"]); target=float(d["+2R"])
        periods=[(f"{h}営業日",ep+h-1) for h in HORIZONS]+[("設定終了日まで",endpos)]
        for label,last in periods:
            if last<ep: continue
            x=_hit(df,ep,stop,target,last); nr=net_r(entry,x[2],risk,comm,slip) if pd.notna(x[2]) and pd.notna(x[4]) else np.nan
            rows.append({"シグナル":d.シグナル,"Stop方式":d.Stop方式,"評価期間":label,"結果":x[0],"結果日":x[1].date() if pd.notna(x[1]) else None,"決済方法":x[3],"Gross_R":x[4],"Net_R":nr,"MFE_R":x[5],"MAE_R":x[6],"期間末Close_R":x[7]})
    return pd.DataFrame(rows)

def position_sizing(design,selected,total,budget,risk_pct,fx,currency,comm,slip):
    rows=[]; name=f"ATR×{float(selected):g}"
    if design.empty:return pd.DataFrame()
    for _,d in design[(design.状態=="R計算可能")&(design.Stop方式==name)].iterrows():
        entry=float(d.Entry_Open); stop=float(d.Stop); allowed=total*risk_pct/100
        buy=entry*(1+slip); sell=stop*(1-slip); per_buy=buy*(1+comm)*fx; per_loss=(buy-sell+buy*comm+sell*comm)*fx
        by_budget=math.floor(budget/per_buy) if per_buy>0 else 0; by_risk=math.floor(allowed/per_loss) if per_loss>0 else 0; shares=max(0,min(by_budget,by_risk)); loss=per_loss*shares
        rows.append({"シグナル":d.シグナル,"採用Stop方式":name,"Entry":entry,"Stop":stop,"1R":d["1R"],"1R_%":d["1R_%"],f"総資金_{currency}":total,f"1銘柄予算_{currency}":budget,"許容損失率_%":risk_pct,f"許容損失額_{currency}":allowed,"予算上の最大株数":by_budget,"損失上限からの最大株数":by_risk,"採用購入株数":shares,f"株価ベース購入額_{currency}":entry*shares*fx,f"Entryコスト込必要額_{currency}":per_buy*shares,f"1株Stop推定損失_{currency}":per_loss,f"Stop時推定総損失_{currency}":loss,"Stop時推定総資金損失率_%":loss/total*100 if total else np.nan,"換算レート":fx})
    return pd.DataFrame(rows)

def money_scenarios(design,ps,selected,fx,currency,comm,slip):
    rows=[]; name=f"ATR×{float(selected):g}"
    if ps.empty:return pd.DataFrame()
    for _,d in design[(design.状態=="R計算可能")&(design.Stop方式==name)].iterrows():
        q=ps[ps.シグナル==d.シグナル]
        if q.empty:continue
        shares=int(q.iloc[0]["採用購入株数"]); entry=float(d.Entry_Open); buy=entry*(1+slip)
        for lab,price in [("Stop (-1R)",float(d.Stop)),("+1R",float(d["+1R"])),("+1.5R",float(d["+1.5R"])),("+2R",float(d["+2R"]))]:
            sell=price*(1-slip); gross=(price-entry)*shares*fx; net=(sell-buy-buy*comm-sell*comm)*shares*fx
            rows.append({"シグナル":d.シグナル,"Stop方式":name,"シナリオ":lab,"価格":price,"購入株数":shares,f"Gross損益_{currency}":gross,f"コスト込推定損益_{currency}":net})
    return pd.DataFrame(rows)

def path_table(df,start,end,d):
    p=df.loc[(df.index>=start)&(df.index<=end)]; base=float(df.loc[d,"Close"])
    if p.empty:return pd.DataFrame()
    return pd.DataFrame({"日付":p.index.date,"Day0区分":["Day0" if x==d else ("前" if x<d else "後") for x in p.index],"Open":p.Open.values,"High":p.High.values,"Low":p.Low.values,"Close":p.Close.values,"BB_Lower":p.BB_Lower.values,"BB_Middle":p.BB_Middle.values,"BB_Upper":p.BB_Upper.values,"ATR":p.ATR.values,"ATR_%":p.ATR_Pct.values,"Day0終値比_%":(p.Close.values/base-1)*100})

def run_case_study(ticker,requested_date,analysis_start,analysis_end,commission=.001,slippage=.001,atr_period=14,atr_multipliers=(1.,1.5,2.),selected_atr_multiplier=1.5,total_capital=1_000_000.,symbol_budget=300_000.,risk_pct=1.,quote_to_capital_fx=1.,capital_currency="JPY"):
    req=pd.Timestamp(requested_date).normalize(); start=pd.Timestamp(analysis_start).normalize(); end=pd.Timestamp(analysis_end).normalize(); today=pd.Timestamp.today().normalize()
    if start>req:return {"error":"分析開始日はBB基準日以前にしてください。"}
    if end<req:return {"error":"分析終了日はBB基準日以降にしてください。"}
    if total_capital<=0 or symbol_budget<=0 or quote_to_capital_fx<=0:return {"error":"総資金・1銘柄予算・換算レートは0より大きい値にしてください。"}
    raw=download_prices(ticker,start-pd.Timedelta(days=550),min(today,end))
    if raw.empty:return {"error":"株価データを取得できませんでした。銘柄コードと日付を確認してください。"}
    df=add_indicators(raw,atr_period); day0,note=resolve_case_date(df,req)
    if day0 is None:return {"error":note}
    starts=df.index[df.index>=start]; ends=df.index[df.index<=min(today,end)]
    if not len(starts) or not len(ends):return {"error":"指定期間の価格データがありません。"}
    actual_start=starts[0]; actual_end=ends[-1]
    design=build_risk_design(df,day0,atr_multipliers)
    ps=position_sizing(design,selected_atr_multiplier,total_capital,symbol_budget,risk_pct,quote_to_capital_fx,capital_currency,commission,slippage)
    scenarios=money_scenarios(design,ps,selected_atr_multiplier,quote_to_capital_fx,capital_currency,commission,slippage)
    visual=design[design.状態=="R計算可能"].copy() if not design.empty else pd.DataFrame()
    if not visual.empty:
        visual["方式"]=visual.シグナル.astype(str)+"｜"+visual.Stop方式.astype(str); visual=visual[["方式","Entry_Open","Stop","1R","1R_%","ATR_シグナル日"]]
    return {"error":None,"ticker":_clean_ticker(ticker),"day0":day0,"date_note":note,"analysis_start":actual_start,"analysis_end":actual_end,"day0_summary":case_summary(df,day0,atr_period),"pre_summary":pre_summary(df,actual_start,day0,atr_period),"signal_window":find_signals(df,day0),"risk_design":design,"position_sizing":ps,"money_scenarios":scenarios,"outcomes":build_outcomes(df,design,actual_end,commission,slippage),"path":path_table(df,actual_start,actual_end,day0),"risk_visual":visual}
