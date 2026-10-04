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




def validation_diagnostics(test):
    rows = []
    for cls in CLASSES:
        predicted = int(test["AI判定"].eq(cls).sum())
        actual = int(test["結果"].eq(cls).sum())
        correct = int((test["AI判定"].eq(cls) & test["結果"].eq(cls)).sum())
        rows.append({"分類": cls, "予測日数": predicted, "実際の日数": actual,
            "正解日数": correct, "予測割合_%": 100 * predicted / len(test) if len(test) else np.nan,
            "予測の的中率_%": 100 * correct / predicted if predicted else np.nan,
            "実際を捉えた割合_%": 100 * correct / actual if actual else np.nan})
    frame = pd.DataFrame(rows)
    lead = frame.loc[frame["予測日数"].eq(frame["予測日数"].max()), "分類"].tolist()
    bias = "・".join(lead) + f" {frame['予測割合_%'].max():.1f}%" if len(test) else "算出不可"
    return frame, bias


def show_validation_diagnostics(bundle):
    frame, bias = validation_diagnostics(bundle["test"])
    up = frame.loc[frame["分類"].eq("上昇")].iloc[0]
    down = frame.loc[frame["分類"].eq("下降")].iloc[0]
    def display(value):
        return f"{value:.1f}%" if pd.notna(value) else "算出不可"
    a,b,c,d = st.columns(4)
    a.metric("AIの正解率", f"{bundle['metrics']['AI正解率']*100:.1f}%")
    b.metric("上昇予測の的中率", display(up["予測の的中率_%"]))
    c.metric("下降を捉えた割合", display(down["実際を捉えた割合_%"]))
    d.metric("予測の偏り（最多分類）", bias)
    st.caption("上昇の的中率＝上昇と予測して当たった日÷上昇予測日数。下降を捉えた割合＝下降と正しく予測した日÷実際の下降日数。偏り＝最も多く予測した分類とその割合。同数は併記します。")
    st.write(f"上昇の的中：{int(up['正解日数'])}／{int(up['予測日数'])}日 ｜ 下降の捕捉：{int(down['正解日数'])}／{int(down['実際の日数'])}日")
    st.caption("予測日数が0の的中率、実際の日数が0の捕捉割合は算出不可です。0%は対象日があるのに正解が0日の場合です。")
    for _,row in frame.iterrows():
        if row["実際の日数"] > 0 and row["予測日数"] == 0:
            st.warning(f"検証期間に実際の『{row['分類']}』は{int(row['実際の日数'])}日ありましたが、AIは一度も『{row['分類']}』と予測していません。")
    with st.expander("予測の偏り・分類ごとの的中率を詳しく確認"):
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption("予測割合だけで良否は決まりません。実際の分類内訳と合わせて確認してください。的中率や捕捉割合は今回の検証期間の実績で、選択日の将来確率ではありません。")



def condition_diagnostics(bundle):
    """Group the fixed model's training and validation rows by exact leaf."""
    model, columns = bundle["model"], bundle["columns"]
    test = bundle["test"].copy()
    train = bundle["train"]
    test["判断条件番号"] = model.apply(test[columns]).astype(int)
    train_leaves = model.apply(train[columns])
    train_predictions = model.predict(train[columns])
    rows = []
    for leaf in sorted(set(train_leaves)):
        mask = train_leaves == leaf
        learning = train.loc[mask]
        validation = test.loc[test["判断条件番号"].eq(leaf)]
        prediction = str(train_predictions[mask][0])
        correct = int(validation["一致"].sum())
        if len(validation):
            explanation = explain_day(bundle, validation.index[0])
            conditions = " / ".join(explanation["rules"].get("条件", pd.Series(dtype=str))) or "特徴による分岐なし"
        else:
            conditions = "検証で該当日なし"
        row = {"判断条件番号": int(leaf), "AI判定": prediction,
            "学習日数": len(learning), "検証日数": len(validation), "検証正解日数": correct,
            "検証的中率_%": 100 * correct / len(validation) if len(validation) else np.nan,
            "検証初日": validation.index.min() if len(validation) else pd.NaT,
            "検証最終日": validation.index.max() if len(validation) else pd.NaT,
            "判断条件": conditions}
        for cls in CLASSES:
            row[f"学習の{cls}日数"] = int(learning["結果"].eq(cls).sum())
            row[f"検証の実際の{cls}日数"] = int(validation["結果"].eq(cls).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    visited = summary.loc[summary["検証日数"].gt(0)]
    if len(visited) == 1:
        message = f"検証{len(test)}日すべてが、同じ判断条件番号 {int(visited.iloc[0]['判断条件番号'])} に入りました。"
    elif visited["AI判定"].nunique() == 1:
        message = f"検証では{len(visited)}種類の判断条件に入りましたが、いずれも『{visited.iloc[0]['AI判定']}』を答える条件でした。"
    else:
        message = f"検証では{len(visited)}種類の判断条件に入り、{visited['AI判定'].nunique()}種類の分類を予測しました。"
    return test, summary, message


def show_condition_diagnostics(bundle, explanation):
    test, summary, message = condition_diagnostics(bundle)
    st.markdown("#### 同じ分類が続く理由：判断条件ごとの検証")
    st.info(message)
    shown = summary.copy()
    shown["検証的中率_%"] = shown["検証的中率_%"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "算出不可")
    for col in ["検証初日", "検証最終日"]:
        shown[col] = shown[col].map(lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else "該当なし")
    shown.insert(0, "選択日の条件", shown["判断条件番号"].eq(explanation["leaf"]).map({True:"★", False:""}))
    st.caption("判断条件番号は、決定木の最終的な条件の組み合わせを識別する番号です。順位・強さ・確率ではありません。設定や学習期間を変えると番号の意味も変わります。★は現在の選択日と同じ条件です。")
    main = ["選択日の条件", "判断条件番号", "AI判定", "学習日数", "検証日数", "検証正解日数", "検証的中率_%"]
    st.dataframe(shown[main], hide_index=True, use_container_width=True)
    st.caption("検証的中率＝この条件で正しく分類した日÷この条件の検証日数。検証0日は算出不可です。学習の割合と検証の的中率を区別してください。")
    with st.expander("条件ごとの学習内訳・検証の実際・分岐条件"):
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.download_button("判断条件ごとの検証CSVを保存", summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="ai_condition_validation.csv", mime="text/csv", key="ai_condition_csv")
    return test


def build_ai_copy_report(result, bundle, explanation, day, sector, include_market, width, atr_period):
    """Plain text snapshot of the current explanation and fixed validation."""
    def table(frame):
        return frame.to_csv(index=False, sep="\t", float_format="%.6f").rstrip()
    def counts(frame, column):
        return ", ".join(f"{c}: {int(frame[column].eq(c).sum())}日" for c in CLASSES)
    diagnostics, bias = validation_diagnostics(bundle["test"])
    condition_test, condition_summary, condition_message = condition_diagnostics(bundle)
    actual = explanation["actual"]
    outcome = "未確定（指定期間内の将来データ不足）" if pd.isna(actual["結果"]) else (
        f"{actual['結果']} / {actual['将来騰落率_%']:+.6f}% / 確定日 {actual['結果確定日'].date()}")
    lines = ["AI分析の確認用レポート v2.6.4",
        "この結果の偏り、検証成績、判定理由を初心者向けに確認してください。",
        f"銘柄: {result['ticker']}",
        f"分析期間: {result['analysis_start']} ～ {result['analysis_end']}",
        f"理由を確認する日: {pd.Timestamp(day).date()}",
        f"判定: {bundle['horizon']}営業日後 / 上昇・下降幅 ±{bundle['threshold']:g}%",
        f"波確認本数: {width} / ATR期間: {atr_period}",
        f"米国市場特徴使用: {include_market} / 所属セクター: {sector or '未選択'}",
        f"株価最終日: {bundle.get('price_asof', '不明')}",
        f"AI分類: {explanation['prediction']} / 判断条件番号: {explanation['leaf']}",
        f"実際の将来結果: {outcome}",
        "\n【検証成績】"]
    lines.extend(f"{k}: {v}" for k,v in bundle["metrics"].items())
    lines += ["正解率などは0～1の値（0.60＝60%）。正解率差も同じ尺度。",
        "\n【的中率・下降の捕捉・予測の偏り】",
        "予測の偏り（最多分類）: " + bias,
        table(diagnostics.fillna("算出不可")),
        "上昇予測の的中率＝上昇正解日数÷上昇予測日数。下降を捉えた割合＝下降正解日数÷実際の下降日数。",
        "的中率の予測日数0・捕捉割合の実際日数0は算出不可。割合は検証実績で将来確率ではありません。",
        "\n【学習・検証の分類内訳】",
        "学習の実際: " + counts(bundle["train"], "結果"),
        "検証の実際: " + counts(bundle["test"], "結果"),
        "検証のAI予測: " + counts(bundle["test"], "AI判定"),
        "\n【検証の混同行列：行＝実際、列＝予測】",
        table(bundle["confusion"].reset_index()),
        "\n【判断条件ごとの検証】", condition_message,
        "判断条件番号はモデル内の識別番号で、順位や確率ではありません。検証0日は的中率を算出できません。",
        table(condition_summary.fillna("該当なし")),
        "\n【選択日の判定理由】",
        table(explanation["rules"]) if not explanation["rules"].empty else "特徴による分岐なし（学習多数派）",
        "\n【選択日の全特徴量】", table(explanation["values"].reset_index(names="日付")),
        "\n【同じ条件の学習例】", table(explanation["distribution"]),
        "\n【検証で役立った特徴：正解率低下・ばらつきの単位はpp】", table(bundle["importance"]),
        "\n【直近20検証日の予測と実際】",
        table(condition_test[["判断条件番号", "AI判定", "結果", "将来騰落率_%", "結果確定日"]].tail(20).reset_index(names="日付")),
        "\n【注意・取得状況】", *bundle.get("notes", []),
        "モデルは検証開始前に固定。学習の未来結果は境界で除外。株価は調整済み。",
        "同じ条件の割合は過去学習例の比率で、将来確率や原因の証明ではありません。"]
    return "\n".join(lines)


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
    show_validation_diagnostics(bundle)
    a,b=st.columns(2)
    a.metric("常に学習多数派を答える正解率",f"{metrics['常に学習多数派を答える正解率']*100:.1f}%")
    b.metric("AIと基準の差",f"{metrics['正解率差']*100:+.1f} pp")
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
    condition_test = show_condition_diagnostics(bundle, explanation)
    report = build_ai_copy_report(result, bundle, explanation, day, sector, include_market, width, atr_period)
    with st.expander("AI結果をコピーして相談する", expanded=True):
        st.caption("下の枠の右上にあるコピーアイコン（重なった四角）を押し、この会話に貼り付けてください。選択日を変えると内容も更新されます。")
        st.code(report, language=None)
        st.download_button("同じAI結果をテキストで保存", report.encode("utf-8-sig"),
            file_name=f"ai_report_{ticker.replace('/', '_')}_{day.strftime('%Y%m%d')}.txt", mime="text/plain", key=f"ai_report_download_{key}")
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
        test=condition_test.reset_index(names="日付")
        st.dataframe(test,hide_index=True,use_container_width=True)
        st.download_button("AI検証結果CSVを保存",test.to_csv(index=False).encode("utf-8-sig"),file_name="ai_validation.csv",mime="text/csv")
    with st.expander("AIに渡した特徴と計算方法"):
        st.dataframe(pd.DataFrame([{"特徴":col,"計算・意味":DEFINITIONS.get(col,"")} for col in bundle["columns"]]),hide_index=True,use_container_width=True)
        st.dataframe(explanation["values"].reset_index(names="日付"),hide_index=True,use_container_width=True)
        st.caption("終値ベースの上昇・下降分類で、Stop・売買コスト・実際の売買利益を予測するモデルではありません。既存のStop設計や資金管理は変更しません。")
