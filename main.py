from datetime import date, timedelta
from io import StringIO
import hashlib
import pandas as pd
import streamlit as st
from case_study_core import (
    APP_VERSION, run_case_study, empty_case_ledger, build_case_ledger_rows,
    merge_case_ledgers, normalize_case_ledger, case_ledger_case_list,
    case_ledger_summary, case_ledger_all_ticker_summary,
    case_ledger_computability_summary, case_ledger_equal_ticker_summary,
    case_ledger_paired_vs_structure, case_ledger_structure_risk_distribution,
    case_ledger_structure_risk_cases, case_ledger_detail,
)

st.set_page_config(page_title="自由銘柄・自由期間 BB下限ケース分析", page_icon="🔎", layout="wide")
st.title("🔎 自由銘柄・自由期間 BB下限ケース分析")
st.caption(f"Version {APP_VERSION} ｜ 1ケース分析を維持＋蓄積ケースの比較診断を強化")
st.info("v5.3前向き検証とは完全に別のケーススタディ用です。AI売買判定は行いません。v2.3ではv2.2の全機能を残したまま、計算可能率・銘柄等重み・同一ケース差・価格構造1R分布を追加します。")

if "case_ledger" not in st.session_state:
    st.session_state.case_ledger=empty_case_ledger()
if "current_result" not in st.session_state:
    st.session_state.current_result=None
if "current_settings" not in st.session_state:
    st.session_state.current_settings=None
if "loaded_ledger_hash" not in st.session_state:
    st.session_state.loaded_ledger_hash=None

with st.expander("v2.3 複数ケース台帳・研究診断の使い方", expanded=True):
    st.write("① 今まで通り1ケースを完全分析します。② 結果下部の『今回のケースを比較台帳に追加』を押します。③ 別の銘柄・別のDay0を分析して追加します。④ 20営業日Net Rを同じ評価基準で比較します。")
    st.write("台帳CSVをダウンロードしておけば、Streamlitを開き直した後も『台帳CSVを読み込む』から続きができます。")
    st.write("v2.2で保存したCSVはそのまま読み込めます。v2.3では銘柄ごとのケース数が違う影響や、価格構造StopとATR Stopの同一ケース差も確認できます。")
    st.warning("比較件数が少ない段階の平均Rは結論ではありません。Stop方式やATR倍率を自動採用する機能ではありません。0.25 / 0.50 / 1.00 ATRの区分も診断表示だけです。")

st.subheader("0. 複数ケース比較台帳")
u1,u2=st.columns([2,1])
with u1:
    uploaded=st.file_uploader("以前保存したケース台帳CSVを読み込む（任意）",type=["csv"])
with u2:
    if st.button("台帳を空にする",use_container_width=True):
        st.session_state.case_ledger=empty_case_ledger(); st.session_state.loaded_ledger_hash=None
        st.success("比較台帳を空にしました。")
if uploaded is not None:
    raw=uploaded.getvalue(); h=hashlib.sha256(raw).hexdigest()
    if h!=st.session_state.loaded_ledger_hash:
        try:
            incoming=pd.read_csv(StringIO(raw.decode("utf-8-sig")))
            st.session_state.case_ledger=merge_case_ledgers(st.session_state.case_ledger,incoming)
            st.session_state.loaded_ledger_hash=h
            st.success(f"台帳CSVを読み込みました。現在 {st.session_state.case_ledger['Case_ID'].nunique()} ケースです。")
        except Exception as e:
            st.error(f"台帳CSVを読み込めませんでした: {e}")

ledger=normalize_case_ledger(st.session_state.case_ledger)
case_count=ledger["Case_ID"].nunique() if not ledger.empty else 0
ticker_count=ledger["銘柄"].nunique() if not ledger.empty else 0
st.write(f"現在の比較台帳: **{case_count}ケース / {ticker_count}銘柄**")
if not ledger.empty:
    st.download_button("比較台帳CSVを保存",data=ledger.to_csv(index=False,float_format="%.6f").encode("utf-8-sig"),file_name="bb_case_ledger_v2_3.csv",mime="text/csv",use_container_width=True)

st.divider()
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
    st.session_state.current_result=result
    st.session_state.current_settings={"commission":commission_pct/100,"slippage":slippage_pct/100}

result=st.session_state.current_result
if result:
    st.success(f"{result['ticker']} ｜ Day0 {result['day0'].date()} ｜ 分析期間 {result['analysis_start'].date()} ～ {result['analysis_end'].date()}")
    if result["date_note"]!="入力日を使用": st.warning(result["date_note"])

    st.subheader("5. Day0と前後環境")
    section(1,"ケース日・BB下限位置・ATR・市場状態",result["day0_summary"],True)
    section(2,"基準日前の環境サマリー",result["pre_summary"],False)
    st.subheader("6. Day0～Day3 シグナル")
    section(3,"4営業日シグナル監査",result["signal_window"],True)
    st.subheader("7. 価格構造StopとATR型1R")
    section(4,"Stop・1R比較",result["risk_design"],True)
    diag=result["risk_diagnostic"]
    if diag is not None and not diag.empty:
        bad=diag[diag["1R診断"].isin(["極端に狭い（価格構造1R<0.25ATR）","狭い（価格構造1R<0.50ATR）"])]
        if not bad.empty: st.warning("価格構造Stopの1RがATRに対して非常に狭いケースがあります。これは自動除外ではなく、確認用の警告です。")
    section(5,"価格構造1R・ATR比診断",diag,True)
    st.subheader("8. 資金管理・購入株数")
    section(6,"予算・許容損失から購入株数を計算",result["position_sizing"],True)
    section(7,"購入後のStop・Target金額損益",result["money_scenarios"],True)
    st.subheader("9. 実際の基準日後の結果")
    section(8,"Stop方式別・実際の結果",result["outcomes"],True)
    st.subheader("10. 基準日前後の価格経路")
    path=result["path"]
    if path is not None and not path.empty: st.line_chart(path.set_index(pd.to_datetime(path["日付"]))[["Close","BB_Lower","BB_Middle","BB_Upper"]],use_container_width=True)
    section(9,"設定期間の価格経路",path,False)
    st.subheader("11. 1R幅の視覚比較")
    visual=result["risk_visual"]
    if visual is not None and not visual.empty: st.bar_chart(visual.set_index("方式")[["1R_%"]],use_container_width=True)
    section(10,"1R幅・ATR換算比較",visual,False)

    st.divider(); st.subheader("12. 今回のケースを比較台帳へ追加")
    st.write("このボタンでは売買ルールを変更せず、今回の分析結果を比較用CSV台帳へ追加するだけです。同じ銘柄・同じDay0を再追加した場合は最新結果で置き換えます。")
    if st.button("今回のケースを比較台帳に追加",type="primary",use_container_width=True):
        settings=st.session_state.current_settings or {"commission":0.001,"slippage":0.001}
        new_rows=build_case_ledger_rows(result,settings["commission"],settings["slippage"])
        st.session_state.case_ledger=merge_case_ledgers(st.session_state.case_ledger,new_rows)
        st.success(f"{result['ticker']} / Day0 {result['day0'].date()} を追加しました。現在 {st.session_state.case_ledger['Case_ID'].nunique()} ケースです。")

st.divider(); st.subheader("13. 複数銘柄・複数BB下限ケース比較")
ledger=normalize_case_ledger(st.session_state.case_ledger)
if ledger.empty:
    st.info("まだ比較台帳にケースがありません。1ケースを分析して『今回のケースを比較台帳に追加』を押してください。")
else:
    st.download_button("最新の比較台帳CSVを保存",data=ledger.to_csv(index=False,float_format="%.6f").encode("utf-8-sig"),file_name="bb_case_ledger_v2_3.csv",mime="text/csv",use_container_width=True,key="download_bottom")
    section(11,"蓄積ケース一覧",case_ledger_case_list(ledger),True)
    section(12,"20営業日・銘柄別Stop比較",case_ledger_summary(ledger,"20営業日"),True)
    section(13,"20営業日・全銘柄参考集計",case_ledger_all_ticker_summary(ledger,"20営業日"),False)
    section(14,"蓄積ケース明細",case_ledger_detail(ledger),False)

    st.subheader("14. v2.3 蓄積ケース研究診断")
    st.caption("以下は売買条件の自動採用ではなく、蓄積データの偏り・Stop差・1R幅を確認する研究診断です。")
    section(15,"20営業日・Net R計算可能率監査",case_ledger_computability_summary(ledger,"20営業日"),True)
    section(16,"20営業日・銘柄等重み参考集計",case_ledger_equal_ticker_summary(ledger,"20営業日"),True)
    section(17,"20営業日・価格構造Stopとの差（同一ケース）",case_ledger_paired_vs_structure(ledger,"20営業日"),True)
    section(18,"価格構造1R・ATR倍率分布",case_ledger_structure_risk_distribution(ledger),True)
    section(19,"価格構造1R・ATR比ケース明細",case_ledger_structure_risk_cases(ledger),False)

    st.info("【v2.3の読み方】16番は各銘柄を同じ1票で平均します。17番は同じCase_ID・同じシグナル内だけでATR方式と価格構造を比較します。18・19番のATR比区分は診断であり、除外条件ではありません。")
    st.warning("【未採用】蓄積結果で平均Rが高いStop方式を、そのまま採用しません。ケース数、銘柄差、時期差、未決着、同日順序不明を確認し、必要なら別の時系列検証へ進みます。")

st.warning("これは過去ケースの研究・資金管理シミュレーションです。売買推奨ではありません。価格構造1R診断は警告表示であり、自動的なStop変更・除外条件ではありません。v5.3凍結AIには後付けしません。")
