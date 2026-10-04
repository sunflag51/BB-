"""Streamlit UI for interpretable AI with chronological held-out verification."""
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from ai_analysis import analyze, explain_day, CLASSES, DEFINITIONS
from market_view import download_market_prices, calculate_market, style, SECTORS

@st.cache_data(ttl=600,show_spinner=False)
def download_ai_prices(ticker,start,end):
    # Exclude the current calendar day: its daily candle may still be forming.
    today = pd.Timestamp.now(tz="Asia/Tokyo").normalize().tz_localize(None)
    cutoff = min(pd.Timestamp(end),today-pd.Timedelta(days=1))
    try:
        p = yf.download(ticker,start=(pd.Timestamp(start)-pd.Timedelta(days=650)).strftime("%Y-%m-%d"),
            end=(cutoff+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),auto_adjust=True,
            interval="1d",progress=False,multi_level_index=False,timeout=10)
        if p is None or p.empty:return pd.DataFrame()
        if isinstance(p.columns,pd.MultiIndex):p.columns=p.columns.get_level_values(0)
        p = p.copy()
        p.index = pd.to_datetime(p.index)
        if p.index.tz is not None:p.index=p.index.tz_localize(None)
        p.index=p.index.normalize()
        for col in ["Open","High","Low","Close","Volume"]:
            if col in p:p[col]=pd.to_numeric(p[col],errors="coerce")
        p=p[~p.index.duplicated(keep="last")].sort_index().loc[:cutoff]
        p=p.dropna(subset=["Open","High","Low","Close"])
        return p[p.Close.gt(0)&p.Open.gt(0)]
    except Exception:
        return pd.DataFrame()


def build_context(result,sector):
    start,end=result["analysis_start"],result["analysis_end"]
    prices,missing=download_market_prices(start,end)
    data=calculate_market(prices,pd.Timestamp(start)-pd.Timedelta(days=10),end,20)
    if data["error"]:return pd.DataFrame(),[data["error"]]
    x=pd.DataFrame({"前日米国4指数平均_%":data["score"]})
    if sector:
        x["前日所属セクター_%"]=data["sectors"][sector]
        rank=data["sectors"].rank(axis=1,ascending=False,method="min")
        x["前日セクター順位"]=rank[sector].where(data["sectors"].notna().all(axis=1))
        x["前日セクターSPY差_pp"]=data["relative"][sector]
    notes=["取得不足の市場データ: "+", ".join(missing)] if missing else []
    return x,notes


def descriptive_view(bundle):
    x=bundle.get("features",pd.DataFrame())
    targets=bundle.get("targets",pd.DataFrame())
    if x.empty or targets.empty:return
    labeled=x.join(targets[["結果"]]).dropna(subset=["結果"])
    st.markdown("#### 選択期間の上昇・下降・横ばいの特徴比較")
    st.caption("選択期間全体を結果別に分けた事後集計です。AIの検証成績とは別です。")
    counts=labeled["結果"].value_counts().reindex(CLASSES,fill_value=0)
    st.dataframe(pd.DataFrame({"分類":CLASSES,"営業日数":counts.values}),hide_index=True,use_container_width=True)
    if labeled.empty:return
    feature=st.selectbox("比較する特徴",list(x.columns),key="ai_compare_feature")
    average=labeled.groupby("結果")[feature].mean().reindex(CLASSES)
    fig=go.Figure(go.Bar(x=average.values,y=CLASSES,orientation="h",
        marker_color=["#ef5350","#777777","#26a69a"],text=[f"{v:.3f}" if pd.notna(v) else "" for v in average],
        textposition="outside",cliponaxis=False))
    style(fig,320)
    v=average.dropna()
    if not v.empty:
        low,high=min(0,float(v.min())),max(0,float(v.max()));span=max(high-low,.1)
        fig.update_xaxes(range=[low-.2*span,high+.2*span],title=feature)
    st.plotly_chart(fig,theme=None,use_container_width=True)
    st.write(DEFINITIONS.get(feature,""))


def show_ai_panel(result):
    st.subheader("AI特徴分析：判断の理由を確認する")
    st.caption("選択した銘柄・分析期間の各営業日を分析します。BBイベントの日だけに限定しません。調整済み株価を使うため、従来の価格チャートと株価が異なる場合があります。")
    a,b,c=st.columns(3)
    horizon=a.selectbox("何営業日後を判定するか",[1,3,5,10,20],index=2,key="ai_horizon")
    threshold=b.number_input("上昇・下降の判定幅（%）",min_value=.1,max_value=30.,value=2.,step=.1,key="ai_threshold")
    width=c.slider("AI用の波の確認本数（前後）",1,10,3,key="ai_wave_width")
    include_market=st.checkbox("前取引日の米国市場・所属セクターも特徴に使う",value=True,key="ai_market_features")
    ticker=result["ticker"]
    sector=st.session_state.get(f"chosen_sector_{ticker}") or None
    if include_market:
        st.caption("米国市場・セクターは20営業日の指標を使用します。銘柄の判定日より前の米国データのみ参照。日本株でも米国市場の参考値です。")
        if sector:st.caption(f"所属セクター：{SECTORS[sector][0]} ({sector})。現在の分類を期間全体に適用します。")
    st.write(f"終値から{horizon}営業日後の終値を比較：＋{threshold:g}%以上＝上昇、−{threshold:g}%以下＝下降、その間＝横ばい。期間末尾の未確定日は学習・検証から除外します。")
    atr_period=int(result.get("atr_period",14))
    signature=str((ticker,str(result["analysis_start"]),str(result["analysis_end"]),horizon,threshold,width,include_market,sector,atr_period))
    key=hashlib.sha256(signature.encode()).hexdigest()[:16]
    if st.button("この条件でAI特徴分析を実行",type="primary",key="run_explainable_ai"):
        with st.spinner("AI用データを取得し、特徴・学習・別期間検証を計算しています…"):
            prices=download_ai_prices(ticker,result["analysis_start"],result["analysis_end"])
            if prices.empty:
                bundle={"error":"AI用の調整済み株価を取得できませんでした。銘柄と期間を確認して再実行してください。"}
            else:
                context,notes=build_context(result,sector) if include_market else (None,[])
                bundle=analyze(prices,result["analysis_start"],result["analysis_end"],horizon,threshold,atr_period,width,context)
                bundle["notes"]=bundle.get("notes",[])+notes
                bundle["price_asof"]=prices.index[-1]
                bundle["prices"]=prices
        st.session_state.ai_bundle={"signature":signature,"result":bundle}
    saved=st.session_state.get("ai_bundle")
    if not saved or saved["signature"]!=signature:
        st.info("条件を選び、AI特徴分析を実行してください。短い期間では検証不足になるため、必要に応じて分析開始日を早めて完全分析をやり直してください。")
        return
    bundle=saved["result"]
    for note in bundle.get("notes",[]):st.info(note)
    if "price_asof" in bundle:st.caption(f"AI用株価の最終日：{bundle['price_asof'].date()}。当日の未完成日足は除外します。")
    if bundle.get("error"):
        st.warning(bundle["error"])
        descriptive_view(bundle)
        return
    metrics=bundle["metrics"]
    st.markdown("#### 別期間での検証成績")
    a,b,c=st.columns(3)
    a.metric("AIの正解率",f"{metrics['AI正解率']*100:.1f}%")
    b.metric("常に学習多数派を答える正解率",f"{metrics['常に学習多数派を答える正解率']*100:.1f}%")
    c.metric("AIと基準の差",f"{metrics['正解率差']*100:+.1f} pp")
    if metrics["正解率差"]<=0:
        st.warning("この検証期間では、AIは単純な多数派の回答を上回っていません。判断理由は参考として確認してください。")
    st.caption("前半約75%で学習、後半約25%で検証。モデルは検証開始前に固定し、検証期間で学習し直しません。日数比は未来結果の除外で変わります。")
    st.write(f"学習 {metrics['学習日数']}日：{metrics['学習開始'].date()}～{metrics['学習最終特徴日'].date()} ／ 検証 {metrics['検証日数']}日：{metrics['検証開始'].date()}～{metrics['検証終了'].date()}")
    with st.expander("未来データ除外・分類ごとの検証を確認"):
        st.write(f"学習の最終結果確定日：{metrics['学習最終結果確定日'].date()}。検証開始：{metrics['検証開始'].date()}。境界から{metrics['除外した境界日数']}日を除外。")
        st.dataframe(pd.DataFrame([{"指標":k,"値":v} for k,v in metrics.items() if k in ["分類を均等に見た正解率","Macro_F1"]]),hide_index=True,use_container_width=True)
        st.write("行＝実際の結果、列＝AIの判定。対角線が正解です。")
        st.dataframe(bundle["confusion"],use_container_width=True)
        for cls in CLASSES:
            if cls not in bundle["model"].classes_:st.info(f"学習期間に『{cls}』の例がなく、この分類を学習できていません。")
        st.caption("同じ期間・設定で繰り返し試すと、検証結果に合わせた調整になります。隣り合う日の将来期間は重なるため、各日を独立した試行とは扱えません。")
    dates=bundle["features"].index[bundle["features"].index>=bundle["test_start"]].tolist()
    st.markdown("#### この日の判断は、なぜそうなったのか")
    day=st.selectbox("理由を確認する日（学習後の期間）",dates,index=len(dates)-1,
        format_func=lambda d:d.strftime("%Y-%m-%d"),key=f"ai_reason_date_{key}")
    explanation=explain_day(bundle,day)
    st.success(f"{ticker}｜{day.date()}の特徴からのAI分類：{explanation['prediction']}")
    actual=explanation["actual"]
    if pd.isna(actual["結果"]):st.write("実際の結果：指定期間内に将来データが足りないため未確定です。")
    else:st.write(f"実際の{horizon}営業日後：{actual['結果']}（{actual['将来騰落率_%']:+.2f}%）、確定日 {actual['結果確定日'].date()}")
    recent=bundle["prices"].loc[:day].tail(60)
    fig=go.Figure(go.Candlestick(x=recent.index.strftime("%Y-%m-%d"),open=recent.Open,
        high=recent.High,low=recent.Low,close=recent.Close,name="判断日までの株価",
        increasing_line_color="#ef5350",increasing_fillcolor="#ef5350",
        decreasing_line_color="#26a69a",decreasing_fillcolor="#26a69a"))
    for period,color in [(50,"#7b1fa2"),(200,"#e65100")]:
        average=bundle["prices"].Close.rolling(period).mean().reindex(recent.index)
        fig.add_trace(go.Scatter(x=recent.index.strftime("%Y-%m-%d"),y=average,
            name=f"{period}日線",mode="lines",line=dict(color=color)))
    style(fig,400);fig.update_xaxes(type="category",nticks=8,rangeslider_visible=False)
    fig.update_layout(title=f"判断日 {day.date()} までの直近60営業日（調整済み株価）")
    st.plotly_chart(fig,theme=None,use_container_width=True)
    st.write("決定木が確認した条件を、判断した順番に表示します。")
    rules=explanation["rules"]
    if rules.empty:st.info("このモデルは特徴で分岐せず、学習多数派で判断しています。")
    else:
        for _,rule in rules.iterrows():
            st.write(f"{int(rule['順番'])}. **{rule['条件']}** ／ 当日の値：{rule['当日の値']:.4f}")
            st.caption(rule["特徴の意味"])
    st.markdown("**同じ判断条件に当てはまった学習例の内訳**")
    st.dataframe(explanation["distribution"],hide_index=True,use_container_width=True)
    st.caption("割合は決定木の同じ条件に入った過去の学習例の割合です。将来にその割合で当たると保証された確率ではありません。原因を証明する説明でもありません。")
    with st.expander("判断の根拠になった過去の日を確認"):
        cols=["結果","将来騰落率_%","結果確定日"]
        if not rules.empty:cols=list(dict.fromkeys(list(rules["確認した特徴"])+cols))
        st.dataframe(explanation["similar"][cols].reset_index(names="日付"),hide_index=True,use_container_width=True)
        st.download_button("同じ判断条件の過去ケースCSV",explanation["similar"].reset_index(names="日付").to_csv(index=False).encode("utf-8-sig"),file_name="ai_rule_cases.csv",mime="text/csv")
    st.markdown("#### 検証で役立った特徴の順位")
    importance=bundle["importance"]
    fig=go.Figure(go.Bar(x=importance["正解率低下_pp"],y=importance["特徴"],orientation="h",
        marker_color="#1565c0",text=[f"{v:+.2f}" for v in importance["正解率低下_pp"]],textposition="outside",cliponaxis=False))
    style(fig,max(420,len(importance)*40+100));fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="特徴の値を入れ替えたときの正解率低下（pp）")
    fig.update_layout(margin=dict(l=215,r=70,t=30,b=60))
    v=importance["正解率低下_pp"];lo,hi=min(0,float(v.min())),max(0,float(v.max()));span=max(hi-lo,1)
    fig.update_xaxes(range=[lo-.2*span,hi+.2*span])
    st.plotly_chart(fig,theme=None,use_container_width=True)
    st.caption("検証データで特徴を入れ替え、正解率がどれだけ下がるかを測定。0付近は効果未確認、負は入れ替えて改善した特徴です。似た特徴同士では重要度が分散する場合があります。")
    st.dataframe(importance,hide_index=True,use_container_width=True)
    descriptive_view(bundle)
    with st.expander("全検証日と予測を確認・保存"):
        test=bundle["test"].reset_index(names="日付")
        st.dataframe(test,hide_index=True,use_container_width=True)
        st.download_button("AI検証結果CSVを保存",test.to_csv(index=False).encode("utf-8-sig"),file_name="ai_validation.csv",mime="text/csv")
    with st.expander("AIに渡した特徴と計算方法"):
        st.dataframe(pd.DataFrame([{"特徴":col,"計算・意味":DEFINITIONS.get(col,"")} for col in bundle["columns"]]),hide_index=True,use_container_width=True)
        st.dataframe(explanation["values"].reset_index(names="日付"),hide_index=True,use_container_width=True)
        st.caption("終値ベースの上昇・下降分類で、Stop・売買コスト・実際の売買利益を予測するモデルではありません。既存のStop設計や資金管理は変更しません。")
