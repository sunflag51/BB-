from datetime import date, timedelta
import pandas as pd
import streamlit as st
from case_study_core import APP_VERSION, run_case_study

st.set_page_config(page_title="自由銘柄・自由期間 BB下限ケース分析", page_icon="🔎", layout="wide")
st.title("🔎 自由銘柄・自由期間 BB下限ケース分析")
st.caption(f"Version {APP_VERSION} ｜ 前後期間＋ATR型1R＋資金管理＋購入株数＋金額損益")
st.info("v5.3前向き検証とは完全に別のケーススタディ用です。AI売買判定は行いません。ATRはシグナル確定時点までのデータだけで計算します。")

with st.expander("今回の完全版でできること", expanded=False):
    st.write("分析開始日・BB基準日・分析終了日を自由入力し、価格構造StopとATR型Stopを比較します。総資金・1銘柄予算・許容損失から購入株数を計算し、Stop/+1R/+1.5R/+2Rの金額損益と実際の値動きを確認します。")
    st.warning("ATR倍率は正解として固定しません。Stop注文はギャップ等で指定価格より不利に約定することがあり、最大損失額は保証値ではありません。")

st.subheader("1. 銘柄と分析期間")
c1,c2=st.columns(2)
with c1: ticker=st.text_input("銘柄コード", value="COST", help="例: COST / AAPL / NVDA / 6857.T")
with c2: case_date=st.date_input("BB下限付近の基準日（Day0）", value=date.today()-timedelta(days=30))
c3,c4=st.columns(2)
with c3: analysis_start=st.date_input("基準日前の分析開始日", value=date.today()-timedelta(days=120))
with c4: analysis_end=st.date_input("基準日後の分析終了日", value=date.today())

st.subheader("2. 1R・Stopの比較条件")
c5,c6,c7=st.columns(3)
with c5: atr_period=st.number_input("ATR期間",5,100,14,1,help="ATR = Average True Range。直近の実際の値幅を使うボラティリティ指標です。")
with c6: atr_multipliers_text=st.text_input("比較するATR倍率",value="1.0,1.5,2.0",help="カンマ区切り。例: 1.0,1.5,2.0,2.5")
with c7: selected_atr_multiplier=st.number_input("資金管理で主に使うATR倍率",0.1,10.0,1.5,0.1,format="%.1f")

st.subheader("3. 資金管理")
st.caption("総資金・予算は資金通貨で入力します。米国株を円資金で見る場合は、換算レートに『1 USD = 何円』を入力します。日本株を円で見る場合は1.0です。")
c8,c9,c10=st.columns(3)
with c8: total_capital=st.number_input("総資金",min_value=0.0,value=1000000.0,step=10000.0,format="%.2f")
with c9: symbol_budget=st.number_input("この1銘柄に使える予算",min_value=0.0,value=300000.0,step=10000.0,format="%.2f")
with c10: risk_pct=st.number_input("1取引の許容損失（総資金に対する%）",0.01,100.0,1.0,0.1,format="%.2f")
c11,c12=st.columns(2)
with c11: quote_to_capital_fx=st.number_input("株価通貨→資金通貨 換算レート",min_value=0.000001,value=1.0,step=0.1,format="%.4f",help="例: COSTがUSD、資金をJPYで見るなら1 USD=150 JPYのとき150。日本株JPYなら1。")
with c12: capital_currency=st.text_input("資金通貨の表示名",value="JPY",help="例: JPY / USD。表示用です。")

st.subheader("4. 売買コスト")
c13,c14=st.columns(2)
with c13: commission_pct=st.number_input("手数料率（片道・%）",0.0,5.0,0.10,0.01,format="%.2f")
with c14: slippage_pct=st.number_input("スリッページ率（片道・%）",0.0,5.0,0.10,0.01,format="%.2f")
run=st.button("この条件で完全分析",type="primary",use_container_width=True)

def csv_text(title,df):
    if df is None or df.empty: return f"【{title}】\n表示対象がありません。"
    return f"【{title}】\n"+df.to_csv(index=False,float_format="%.4f").rstrip()

def section(num,title,df,expanded=False):
    with st.expander(f"{num} {title}",expanded=expanded):
        if df is None or df.empty: st.write("表示対象がありません。")
        else: st.dataframe(df.round(4),use_container_width=True,hide_index=True)
        st.code(csv_text(f"{num} {title}",df),language=None)

def parse_multipliers(text):
    vals=[]
    for x in str(text).split(","):
        try:
            v=float(x.strip())
            if v>0: vals.append(v)
        except: pass
    vals.append(float(selected_atr_multiplier))
    return sorted(set(round(v,4) for v in vals))

if run:
    if analysis_start>case_date: st.error("分析開始日はBB基準日以前にしてください。"); st.stop()
    if analysis_end<case_date: st.error("分析終了日はBB基準日以降にしてください。"); st.stop()
    if symbol_budget>total_capital and total_capital>0: st.warning("1銘柄予算が総資金を上回っています。株数計算では入力された1銘柄予算を使用します。")
    multipliers=parse_multipliers(atr_multipliers_text)
    with st.spinner("株価データを取得し、前後期間・ATR・資金管理まで計算しています..."):
        result=run_case_study(ticker,case_date,analysis_start,analysis_end,commission_pct/100,slippage_pct/100,int(atr_period),multipliers,float(selected_atr_multiplier),float(total_capital),float(symbol_budget),float(risk_pct),float(quote_to_capital_fx),capital_currency.strip() or "資金通貨")
    if result.get("error"): st.error(result["error"]); st.stop()
    st.success(f"{result['ticker']} ｜ Day0 {result['day0'].date()} ｜ 分析期間 {result['analysis_start'].date()} ～ {result['analysis_end'].date()}")
    if result["date_note"]!="入力日を使用": st.warning(result["date_note"])
    st.subheader("5. Day0と前後環境")
    section(1,"ケース日・BB下限位置・ATR・市場状態",result["day0_summary"],True)
    section(2,"基準日前の環境サマリー",result["pre_summary"],False)
    st.subheader("6. Day0～Day3 シグナル")
    section(3,"4営業日シグナル監査",result["signal_window"],True)
    st.subheader("7. 価格構造StopとATR型1R")
    section(4,"Stop・1R比較",result["risk_design"],True)
    st.subheader("8. 資金管理・購入株数")
    section(5,"予算・許容損失から購入株数を計算",result["position_sizing"],True)
    section(6,"購入後のStop・Target金額損益",result["money_scenarios"],True)
    st.subheader("9. 実際の基準日後の結果")
    section(7,"Stop方式別・実際の結果",result["outcomes"],True)
    st.subheader("10. 基準日前後の価格経路")
    path=result["path"]
    if path is not None and not path.empty:
        st.line_chart(path.set_index(pd.to_datetime(path["日付"]))[["Close","BB_Lower","BB_Middle","BB_Upper"]],use_container_width=True)
    section(8,"設定期間の価格経路",path,False)
    st.subheader("11. 1R幅の視覚比較")
    visual=result["risk_visual"]
    if visual is not None and not visual.empty: st.bar_chart(visual.set_index("方式")[["1R_%"]],use_container_width=True)
    section(9,"1R幅・ATR換算比較",visual,False)
    st.warning("これは過去ケースの研究・資金管理シミュレーションです。売買推奨ではありません。ATR倍率や許容損失率の適切さは別途検証が必要です。ギャップ、流動性、税金、為替変動などにより実際の損益は異なります。v5.3凍結AIには後付けしません。")
else:
    st.caption("条件を入力して『この条件で完全分析』を押してください。")
