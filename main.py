from datetime import date, timedelta
import pandas as pd
import streamlit as st

from case_study_core import APP_VERSION, run_case_study

st.set_page_config(page_title="自由銘柄・自由日付 BB下限ケース分析", page_icon="🔎", layout="wide")
st.title("🔎 自由銘柄・自由日付 BB下限ケース分析")
st.caption(f"Version {APP_VERSION} ｜ GOOG/NVDA研究で固定した考え方を使う独立ケーススタディ用プログラム")
st.info("このプログラムは v5.3 前向き検証とは完全に別です。入力した銘柄・日付を研究指標で調査します。AIの買い判定は行いません。")

with st.expander("このプログラムで固定している研究ルール", expanded=False):
    st.write("ケース日をDay0としてDay0～Day3を観察します。下落停止＝Higher Low ＋ Close Up、反発開始＝終値が前日高値を上回る、Entry＝シグナル翌営業日Open、Stop＝Day0からシグナル日までの最安値、Target＝+2R、評価＝5/10/20営業日です。")
    st.write("Gap、同日Stop/Target順序不明、期間末Close、手数料・スリッページを既存研究と同じ考え方で扱います。")
    st.warning("入力日は『研究したいケース日』です。BB下限への到達を強制条件にはしません。実際のBB下限との距離を表示するので、選んだ日がどの程度下限付近だったかも同時に確認できます。")

st.subheader("1. 調査条件を入力")
c1, c2 = st.columns(2)
with c1:
    ticker = st.text_input("銘柄コード", value="NVDA", help="例: AAPL / MSFT / META / TSLA / 6857.T")
with c2:
    case_date = st.date_input("BB下限付近として調べたい日", value=date.today() - timedelta(days=90))

c3, c4 = st.columns(2)
with c3:
    commission_pct = st.number_input("手数料率（片道・%）", min_value=0.0, max_value=5.0, value=0.10, step=0.01, format="%.2f")
with c4:
    slippage_pct = st.number_input("スリッページ率（片道・%）", min_value=0.0, max_value=5.0, value=0.10, step=0.01, format="%.2f")

run = st.button("この銘柄・日付を分析", type="primary", use_container_width=True)

def csv_text(title, df):
    if df is None or df.empty:
        return f"【{title}】\n表示対象がありません。"
    return f"【{title}】\n" + df.to_csv(index=False, float_format="%.4f").rstrip()

def section(num, title, df, expanded=False):
    with st.expander(f"{num} {title}", expanded=expanded):
        if df is None or df.empty:
            st.write("表示対象がありません。")
        else:
            st.dataframe(df.round(4), use_container_width=True, hide_index=True)
        st.code(csv_text(f"{num} {title}", df), language=None)

if run:
    with st.spinner("株価データを取得してケース分析しています..."):
        result = run_case_study(
            ticker=ticker,
            requested_date=case_date,
            commission=commission_pct/100.0,
            slippage=slippage_pct/100.0,
        )

    if result.get("error"):
        st.error(result["error"])
        st.stop()

    st.success(f"{result['ticker']} ｜ ケース日 {result['day0'].date()} の分析完了")
    if result["date_note"] != "入力日を使用":
        st.warning(result["date_note"])

    st.subheader("2. ケース日そのものを確認")
    section(1, "ケース日・BB下限位置と市場状態", result["day0_summary"], True)

    st.subheader("3. Day0～Day3でシグナルを確認")
    section(2, "4営業日シグナル監査", result["signal_window"], True)
    section(3, "下落停止・反発開始 R設計", result["design"], True)

    st.subheader("4. 入力日以降の結果")
    section(4, "5・10・20営業日 2R先着・Net R・MFE/MAE", result["outcomes"], True)

    st.subheader("5. 入力日以降の価格経路")
    path = result["path"]
    if path is not None and not path.empty:
        chart = path.set_index(pd.to_datetime(path["日付"]))[["Close", "BB_Lower", "BB_Middle", "BB_Upper"]]
        st.line_chart(chart, use_container_width=True)
    section(5, "25営業日価格経路", path, False)

    st.subheader("6. 読み方")
    st.write("まず1番で、入力日が本当にBB下限付近だったかを確認します。次に2番でDay0～Day3の下落停止・反発開始を確認し、3番でEntry・Stop・1R・+2Rを確認します。最後に4番で5/10/20営業日のTarget/Stop先着、Net R、MFE/MAEを比較します。")
    st.warning("これは過去ケースを調べる研究ツールです。表示結果は売買推奨ではありません。また、この自由研究の結果をv5.3凍結AIへ後付けしません。")
else:
    st.caption("銘柄コードと日付を入力して「この銘柄・日付を分析」を押してください。")
