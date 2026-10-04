from datetime import date, timedelta
from io import StringIO
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from market_view import show_market_context
from ai_view import show_ai_panel
from ticker_sheet import PRESET_TICKERS, normalize_sheet, add_ticker, remove_tickers, options_with_saved, normalize_ticker
from capital_profiles import normalize_profiles, profile_values, save_profile, delete_profile, PROFILE_FIELDS, conversion_rate
from analysis_profiles import normalize_analysis_profiles, save_analysis_profile, delete_analysis_profile, PROFILE_FIELDS as ANALYSIS_PROFILE_FIELDS
from case_study_core import (
    APP_VERSION, fetch_usd_jpy, confirmed_full_period_swings, run_case_study, empty_case_ledger, build_case_ledger_rows,
    merge_case_ledgers, normalize_case_ledger, case_ledger_case_list,
    case_ledger_summary, case_ledger_all_ticker_summary,
    case_ledger_computability_summary, case_ledger_equal_ticker_summary,
    case_ledger_paired_vs_structure, case_ledger_structure_risk_distribution,
    case_ledger_structure_risk_cases, case_ledger_detail,
)

st.set_page_config(page_title="自由銘柄・自由期間 BB下限ケース分析", page_icon="🔎", layout="wide")

@st.cache_data(ttl=300, show_spinner=False)
def cached_usd_jpy():
    return fetch_usd_jpy()

ANALYSIS_DATE_DEFAULTS={"case_date_value":date.today()-timedelta(days=30),"analysis_start_value":date.today()-timedelta(days=120),"analysis_end_value":date.today()}
for _key,_value in ANALYSIS_DATE_DEFAULTS.items():st.session_state.setdefault(_key,_value)
if "analysis_profiles" not in st.session_state:st.session_state.analysis_profiles=pd.DataFrame(columns=ANALYSIS_PROFILE_FIELDS)

CAPITAL_DEFAULTS={"cap_total":1_000_000.0,"cap_budget":300_000.0,"cap_risk_pct":1.0,
    "cap_currency":"JPY","quote_currency":"USD","capital_fx_mode":"自動（ドル円）","manual_quote_fx":1.0}
for _key,_value in CAPITAL_DEFAULTS.items():st.session_state.setdefault(_key,_value)
if "capital_profiles" not in st.session_state:st.session_state.capital_profiles=pd.DataFrame(columns=PROFILE_FIELDS)

st.title("🔎 自由銘柄・自由期間 BB下限ケース分析")
st.caption(f"Version {APP_VERSION} ｜ ケース分析・市場セクター・判断理由を確認するAI分析")
st.info("このアプリは『何を見ればよいか分からない』を減らすため、表だけでなく、先に見るべきポイントをカード・グラフで表示します。AIは上昇・下降の特徴を分析します。自動売買は行いません。")

if "case_ledger" not in st.session_state:
    st.session_state.case_ledger = empty_case_ledger()
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_settings" not in st.session_state:
    st.session_state.current_settings = None
if "loaded_ledger_hash" not in st.session_state:
    st.session_state.loaded_ledger_hash = None


with st.expander("このアプリの見方（初心者向け）", expanded=True):
    st.write("**まず見る順番は4つだけです。**")
    st.write("① **価格チャート**で『BB下限付近から反発したか』を見る")
    st.write("② **1RとATRの比較**で『Stop幅が狭すぎないか』を見る")
    st.write("③ **20営業日の結果比較**で『どのStop方式が結果として良かったか』を見る")
    st.write("④ ケースを追加したら、**蓄積ケースの比較グラフ**で『銘柄をまたいでも傾向があるか』を見る")
    st.warning("このアプリは『自動で正解を決める』ものではありません。グラフで見やすくする研究用ツールです。")


st.subheader("0. 複数ケース比較台帳")
u1, u2 = st.columns([2, 1])
with u1:
    uploaded = st.file_uploader("以前保存したケース台帳CSVを読み込む（任意）", type=["csv"])
with u2:
    if st.button("台帳を空にする", use_container_width=True):
        st.session_state.case_ledger = empty_case_ledger()
        st.session_state.loaded_ledger_hash = None
        st.success("比較台帳を空にしました。")
if uploaded is not None:
    raw = uploaded.getvalue()
    h = hashlib.sha256(raw).hexdigest()
    if h != st.session_state.loaded_ledger_hash:
        try:
            incoming = pd.read_csv(StringIO(raw.decode("utf-8-sig")))
            st.session_state.case_ledger = merge_case_ledgers(st.session_state.case_ledger, incoming)
            st.session_state.loaded_ledger_hash = h
            st.success(f"台帳CSVを読み込みました。現在 {st.session_state.case_ledger['Case_ID'].nunique()} ケースです。")
        except Exception as e:
            st.error(f"台帳CSVを読み込めませんでした: {e}")

ledger = normalize_case_ledger(st.session_state.case_ledger)
case_count = ledger["Case_ID"].nunique() if not ledger.empty else 0
ticker_count = ledger["銘柄"].nunique() if not ledger.empty else 0
m1, m2 = st.columns(2)
m1.metric("現在の比較ケース数", f"{case_count}")
m2.metric("現在の比較銘柄数", f"{ticker_count}")
if not ledger.empty:
    st.download_button(
        "比較台帳CSVを保存",
        data=ledger.to_csv(index=False, float_format="%.6f").encode("utf-8-sig"),
        file_name="bb_case_ledger_v2_4.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.subheader("銘柄の保存シート")
if "saved_tickers" not in st.session_state:
    st.session_state.saved_tickers = pd.DataFrame(columns=["銘柄コード", "銘柄名"])
st.session_state.saved_tickers = normalize_sheet(st.session_state.saved_tickers)
with st.expander("銘柄を追加・削除・保存／復元", expanded=False):
    st.caption("一覧はこの画面で追加・削除できます。CSVをダウンロードしておくと、次回起動時にも復元できます。")
    saved_now=st.session_state.saved_tickers.copy()
    if not saved_now.empty:
        st.dataframe(saved_now,hide_index=True,use_container_width=True)
        remove_options={f"{row['銘柄コード']}（{row['銘柄名']}）":row['銘柄コード'] for _,row in saved_now.iterrows()}
        remove_labels=["削除する銘柄を選択"]+list(remove_options)
        delete_key="ticker_sheet_remove"
        if st.session_state.get(delete_key) not in remove_labels:st.session_state[delete_key]=remove_labels[0]
        remove_label=st.selectbox("削除する銘柄",remove_labels,key=delete_key)
        if st.button("選んだ銘柄を削除",key="delete_saved_tickers"):
            code=remove_options.get(remove_label)
            if code:
                st.session_state.saved_tickers=remove_tickers(st.session_state.saved_tickers,[code])
                st.rerun()
    else:
        st.info("保存シートは空です。銘柄を選び『選択銘柄を保存シートに追加』を押してください。")
    st.download_button("保存シートCSVをダウンロード",
        st.session_state.saved_tickers.to_csv(index=False).encode("utf-8-sig"),
        file_name="saved_tickers.csv",mime="text/csv",key="download_saved_tickers")
    uploaded_watchlist=st.file_uploader("以前保存した銘柄シートCSVを読み込む",type=["csv"],key="upload_saved_tickers")
    if uploaded_watchlist is not None:
        if st.button("CSVの内容で銘柄シートを復元",key="restore_saved_tickers"):
            try:
                raw_sheet=pd.read_csv(uploaded_watchlist)
                aliases={"ticker":"銘柄コード","symbol":"銘柄コード","code":"銘柄コード","name":"銘柄名"}
                incoming_columns={aliases.get(str(c).strip().casefold(),c) for c in raw_sheet.columns}
                if "銘柄コード" not in incoming_columns:raise ValueError("CSVに『銘柄コード』列がありません。")
                st.session_state.saved_tickers=normalize_sheet(raw_sheet)
                st.rerun()
            except Exception as exc:st.error(f"CSVを読み込めませんでした。銘柄コード・銘柄名の列を確認してください。{exc}")

with st.expander("分析条件の保存・呼び出し", expanded=False):
    st.caption("銘柄と基準日・分析開始日・分析終了日をひとまとめにして保存します。アプリ再起動後も使う場合はCSVを保存して読み込んでください。")
    st.session_state.analysis_profiles = normalize_analysis_profiles(st.session_state.analysis_profiles)
    saved_analysis = st.session_state.analysis_profiles
    if not saved_analysis.empty:
        st.dataframe(saved_analysis, hide_index=True, use_container_width=True)
        profile_choices = saved_analysis["保存名"].tolist()
        load_key = "analysis_profile_load_choice"
        if st.session_state.get(load_key) not in profile_choices:
            st.session_state[load_key] = profile_choices[0]
        load_name = st.selectbox("呼び出す分析条件", profile_choices, key=load_key)
        if st.button("選んだ分析条件を呼び出す", key="load_analysis_profile"):
            row = saved_analysis.loc[saved_analysis["保存名"] == load_name].iloc[-1]
            code = row["銘柄コード"]
            opts = options_with_saved(st.session_state.saved_tickers)
            matching = next((label for label, value in opts.items() if value == code), None)
            if matching:
                st.session_state["ticker_choice"] = matching
            else:
                st.session_state["ticker_choice"] = "その他（銘柄コードを入力）"
                st.session_state["ticker_custom"] = code
                st.session_state["ticker_custom_name"] = row["銘柄名"]
            st.session_state["case_date_value"] = pd.Timestamp(row["基準日"]).date()
            st.session_state["analysis_start_value"] = pd.Timestamp(row["分析開始日"]).date()
            st.session_state["analysis_end_value"] = pd.Timestamp(row["分析終了日"]).date()
            st.session_state.current_result = None
            st.session_state.current_settings = None
            st.rerun()
        delete_key = "analysis_profile_delete_choice"
        if st.session_state.get(delete_key) not in profile_choices:
            st.session_state[delete_key] = profile_choices[0]
        delete_name = st.selectbox("削除する分析条件", profile_choices, key=delete_key)
        if st.button("選んだ分析条件を削除", key="delete_analysis_profile"):
            st.session_state.analysis_profiles = delete_analysis_profile(saved_analysis, delete_name)
            st.rerun()
    else:
        st.info("保存した分析条件はありません。銘柄と期間を入力してから、この下の保存欄で追加できます。")
    st.download_button("分析条件CSVをダウンロード", saved_analysis.to_csv(index=False).encode("utf-8-sig"),
        file_name="analysis_profiles.csv", mime="text/csv", key="download_analysis_profiles")
    uploaded_analysis = st.file_uploader("以前保存した分析条件CSVを読み込む", type=["csv"], key="upload_analysis_profiles")
    if uploaded_analysis is not None and st.button("CSVの内容で分析条件を復元", key="restore_analysis_profiles"):
        try:
            incoming = pd.read_csv(uploaded_analysis)
            restored = normalize_analysis_profiles(incoming)
            if restored.empty and not incoming.empty:
                raise ValueError("有効な保存名・銘柄・3つの日付が見つかりません。")
            st.session_state.analysis_profiles = restored
            st.rerun()
        except Exception as exc:
            st.error(f"CSVを読み込めませんでした: {exc}")

st.subheader("1. 銘柄と分析期間")
c1, c2 = st.columns(2)
with c1:
    ticker_options=options_with_saved(st.session_state.saved_tickers)
    ticker_labels=list(ticker_options)
    current=st.session_state.get("ticker_choice","COST（コストコ）")
    if current not in ticker_options:current="COST（コストコ）"
    if st.session_state.get("ticker_choice") not in ticker_options:
        st.session_state["ticker_choice"]=current
    ticker_choice=st.selectbox("銘柄（会社名付き）",ticker_labels,index=ticker_labels.index(current),key="ticker_choice")
    selected_code=ticker_options[ticker_choice]
    if selected_code=="__CUSTOM__":
        ticker=normalize_ticker(st.text_input("銘柄コードを入力",value="",placeholder="例: AMZN / 7203.T",key="ticker_custom"))
        custom_name=st.text_input("保存シートに表示する会社名（任意）",value="",placeholder="例: Amazon",key="ticker_custom_name")
    else:
        ticker=selected_code
        custom_name=None
    st.caption(f"分析に使うコード：{ticker or '未入力'}")
    if ticker and st.button("選択銘柄を保存シートに追加",key="add_saved_ticker"):
        updated,added=add_ticker(st.session_state.saved_tickers,ticker,custom_name or None)
        st.session_state.saved_tickers=updated
        if added:st.rerun()
        else:st.info("この銘柄は登録済みです。")
with c2:
    case_date = st.date_input("BB下限付近の基準日（Day0）", key="case_date_value")
c3, c4 = st.columns(2)
with c3:
    analysis_start = st.date_input("基準日前の分析開始日", key="analysis_start_value")
with c4:
    analysis_end = st.date_input("分析終了日", key="analysis_end_value")

with st.expander("現在の銘柄・分析期間を保存", expanded=False):
    preset_name = st.text_input("保存名", placeholder="例：NVDA 直近1年", key="analysis_profile_new_name")
    if st.button("現在の銘柄・期間を保存", key="save_analysis_profile"):
        if ticker_options.get(ticker_choice) == "__CUSTOM__":
            profile_ticker_name = custom_name or ticker
        else:
            profile_ticker_name = PRESET_TICKERS.get(ticker, ticker)
            for _, saved_row in st.session_state.saved_tickers.iterrows():
                if saved_row["銘柄コード"] == ticker:
                    profile_ticker_name = saved_row["銘柄名"]
                    break
        try:
            st.session_state.analysis_profiles = save_analysis_profile(
                st.session_state.analysis_profiles, preset_name, ticker, profile_ticker_name,
                case_date, analysis_start, analysis_end)
            st.success(f"「{preset_name.strip()}」を保存しました。")
        except Exception as exc:
            st.error(str(exc))

st.subheader("2. 1R・Stopの比較条件")
c5, c6, c7 = st.columns(3)
with c5:
    atr_period = st.number_input("ATR期間", 5, 100, 14, 1, help="ATR = Average True Range。直近の値幅の大きさです。")
with c6:
    atr_multipliers_text = st.text_input("比較するATR倍率", value="1.0,1.5,2.0", help="カンマ区切り。例: 1.0,1.5,2.0,2.5")
with c7:
    selected_atr_multiplier = st.number_input("資金管理で主に使うATR倍率", 0.1, 10.0, 1.5, 0.1, format="%.1f")

st.subheader("3. 資金管理・保存設定")
st.caption("通貨は選択中の銘柄コードから初期推定します。米国株USD・円資金JPYでは最新取得可能なドル円を自動取得します。日本株JPY・円資金JPYは換算1です。")
# When selecting another ticker, infer its quote currency; allow an explicit override.
if st.session_state.get("quote_currency_ticker") != ticker:
    st.session_state["quote_currency"]="JPY" if ticker.upper().endswith('.T') else "USD"
    st.session_state["quote_currency_ticker"]=ticker

with st.expander("資金管理の初期値を保存・読み込み",expanded=False):
    st.caption("名前を付けて複数の設定を保存できます。設定一覧はCSVでダウンロードし、次回は読み込んで復元します。")
    uploaded_profiles=st.file_uploader("以前保存した資金設定CSV",type=["csv"],key="capital_profiles_upload")
    if uploaded_profiles is not None and st.button("CSVから資金設定を復元",key="capital_profiles_restore"):
        try:
            raw_profiles=pd.read_csv(uploaded_profiles)
            if not set(PROFILE_FIELDS).issubset(raw_profiles.columns):raise ValueError("必要な資金設定の列がありません。")
            st.session_state.capital_profiles=normalize_profiles(raw_profiles)
            st.rerun()
        except Exception as exc:st.error(f"CSVを読み込めませんでした。ファイル内容を確認してください。{exc}")
    st.download_button("資金設定CSVをダウンロード",
        normalize_profiles(st.session_state.capital_profiles).to_csv(index=False).encode("utf-8-sig"),
        file_name="capital_profiles.csv",mime="text/csv",key="capital_profiles_download")
    profiles=normalize_profiles(st.session_state.capital_profiles)
    choices=["読み込む設定を選択"]+profiles["プロファイル名"].tolist()
    if st.session_state.get("capital_profile_choice") not in choices:st.session_state["capital_profile_choice"]=choices[0]
    chosen_profile=st.selectbox("保存した資金設定",choices,key="capital_profile_choice")
    p1,p2=st.columns(2)
    with p1:
        if st.button("選んだ設定を読み込む",key="capital_profile_load"):
            row=profiles.loc[profiles["プロファイル名"].eq(chosen_profile)]
            if not row.empty:
                values=profile_values(row.iloc[0])
                widget_keys={"total_capital":"cap_total","symbol_budget":"cap_budget","risk_pct":"cap_risk_pct",
                    "capital_currency":"cap_currency","quote_currency":"quote_currency",
                    "capital_fx_mode":"capital_fx_mode","manual_quote_fx":"manual_quote_fx"}
                for _source,_widget in widget_keys.items():
                    if _source in values and pd.notna(values[_source]):st.session_state[_widget]=values[_source]
                st.session_state["quote_currency_ticker"]=ticker
                st.rerun()
    with p2:
        delete_choices=["削除する設定を選択"]+profiles["プロファイル名"].tolist()
        if st.session_state.get("capital_profile_delete_choice") not in delete_choices:st.session_state["capital_profile_delete_choice"]=delete_choices[0]
        delete_choice=st.selectbox("削除する保存設定",delete_choices,key="capital_profile_delete_choice")
        if st.button("選んだ設定を削除",key="capital_profile_delete") and delete_choice!=delete_choices[0]:
            st.session_state.capital_profiles=delete_profile(profiles,delete_choice)
            st.rerun()
    st.dataframe(profiles,hide_index=True,use_container_width=True)

cc1,cc2,cc3=st.columns(3)
with cc1:
    quote_currency=st.selectbox("株価の通貨",["USD","JPY"],key="quote_currency")
with cc2:
    capital_currency=st.selectbox("資金通貨",["JPY","USD"],key="cap_currency")
with cc3:
    fx_mode=st.selectbox("通貨換算方法",["自動（ドル円）","手入力"],key="capital_fx_mode")
quote_to_capital_fx=conversion_rate(quote_currency,capital_currency,method=fx_mode,
    manual_rate=st.session_state.get("manual_quote_fx",1.0)) if quote_currency==capital_currency or fx_mode=="手入力" else None
if quote_currency==capital_currency:
    st.info(f"株価通貨と資金通貨が同じため、換算レートは1.0000 {capital_currency}/{quote_currency}です。")
elif fx_mode=="自動（ドル円）" and {quote_currency,capital_currency}=={"USD","JPY"}:
    if st.button("ドル円レートを再取得",key="refresh_capital_fx"):cached_usd_jpy.clear()
    with st.spinner("ドル円レートを取得しています…"):
        quote=cached_usd_jpy()
    if quote.get("error"):
        st.warning(quote["error"]+" 手入力へ切り替えるか、再取得してください。")
    else:
        quote_to_capital_fx=conversion_rate(quote_currency,capital_currency,usd_jpy=quote["rate"])
        st.metric("自動取得した換算レート",f"1 {quote_currency} = {quote_to_capital_fx:.4f} {capital_currency}")
        st.caption(f"配信元データ時刻：{quote['asof']}。最新取得可能な1時間足で、5分間キャッシュします。")
elif quote_currency!=capital_currency and fx_mode=="手入力":
    default_fx=float(st.session_state.get("manual_quote_fx",1.0))
    capital_fx=st.number_input(f"株価通貨→資金通貨（1 {quote_currency} = 何 {capital_currency}）",min_value=.000001,value=default_fx,step=.1,format="%.6f",key="manual_quote_fx")
    quote_to_capital_fx=conversion_rate(quote_currency,capital_currency,manual_rate=capital_fx,method=fx_mode)
else:
    st.error("この通貨の組み合わせは自動換算できません。手入力へ切り替えてください。")

c8,c9,c10=st.columns(3)
with c8:total_capital=st.number_input(f"総資金（{capital_currency}）",min_value=0.,step=10000.,format="%.2f",key="cap_total")
with c9:symbol_budget=st.number_input(f"この1銘柄に使える予算（{capital_currency}）",min_value=0.,step=10000.,format="%.2f",key="cap_budget")
with c10:risk_pct=st.number_input("1取引の許容損失（総資金に対する%）",.01,100.,step=.1,format="%.2f",key="cap_risk_pct")

save1,save2=st.columns([1,2])
with save1:profile_new_name=st.text_input("保存名",placeholder="例：円資金・米国株",key="capital_profile_new_name")
with save2:
    if st.button("現在の資金設定を保存",key="capital_profile_save"):
        values={"total_capital":total_capital,"symbol_budget":symbol_budget,"risk_pct":risk_pct,
            "capital_currency":capital_currency,"quote_currency":quote_currency,
            "capital_fx_mode":fx_mode,"manual_quote_fx":st.session_state.get("manual_quote_fx",1.0)}
        try:
            st.session_state.capital_profiles=save_profile(st.session_state.capital_profiles,profile_new_name,values)
            st.rerun()
        except ValueError as exc:st.error(str(exc))

st.subheader("4. 売買コスト")
c13, c14 = st.columns(2)
with c13:
    commission_pct = st.number_input("手数料率（片道・%）", 0.0, 5.0, 0.10, 0.01, format="%.2f")
with c14:
    slippage_pct = st.number_input("スリッページ率（片道・%）", 0.0, 5.0, 0.10, 0.01, format="%.2f")

run = st.button("この条件で完全分析", type="primary", use_container_width=True)


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


def parse_multipliers(text):
    vals = []
    for x in str(text).split(","):
        try:
            v = float(x.strip())
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    vals.append(float(selected_atr_multiplier))
    return sorted(set(round(v, 4) for v in vals))


def fmt_num(x, digits=2, suffix=""):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}{suffix}"


def first_row(df):
    if df is None or df.empty:
        return None
    return df.iloc[0]


def beginner_signal_summary(result, selected_multiplier, currency):
    diag = result.get("risk_diagnostic", pd.DataFrame())
    pos = result.get("position_sizing", pd.DataFrame())
    outcomes = result.get("outcomes", pd.DataFrame())
    selected_name = f"ATR×{float(selected_multiplier):g}"
    rows = []
    for sig in ["下落停止", "反発開始"]:
        d = diag[diag["シグナル"] == sig] if not diag.empty else pd.DataFrame()
        p = pos[pos["シグナル"] == sig] if not pos.empty else pd.DataFrame()
        o20 = outcomes[(outcomes["シグナル"] == sig) & (outcomes["評価期間"] == "20営業日")] if not outcomes.empty else pd.DataFrame()
        d0 = first_row(d)
        p0 = first_row(p)
        best = first_row(o20.sort_values("Net_R", ascending=False)) if not o20.empty else None
        sel = first_row(o20[o20["Stop方式"] == selected_name]) if not o20.empty else None
        design = result.get("risk_design", pd.DataFrame())
        q = design[design["シグナル"] == sig] if not design.empty else pd.DataFrame()
        window = result.get("signal_window", pd.DataFrame())
        has_signal = not window.empty and sig in window and window[sig].eq(True).any()
        if not has_signal:
            status = "Day0～Day3にシグナル未成立"
            reason = "この条件ではEntry・購入株数・想定損失を計算しません。" if len(window) >= 4 else "観察期間のデータが不足しています。分析終了日を延ばして再分析してください。"
        elif q.empty or not q["状態"].eq("R計算可能").any():
            status = "シグナル成立・EntryまたはStop設計不可"
            reason = " / ".join(q["状態"].dropna().astype(str).unique()) if not q.empty else "設計データなし"
        elif p0 is None:
            status = "シグナル成立・選択ATRの株数計算なし"
            reason = "選択したATR倍率の設計がありません。比較ATR倍率とATR期間を確認して再分析してください。"
        else:
            status = "シグナル成立・株数計算済み"
            reason = "20営業日の結果が未確定の場合は、将来データ不足などを結果欄で確認してください。"
        rows.append({
            "判定状況": status,
            "確認事項": reason,
            "シグナル": sig,
            "価格構造1R_%": d0.get("1R_%") if d0 is not None else np.nan,
            "価格構造1R_ATR倍率": d0.get("1R_ATR倍率") if d0 is not None else np.nan,
            "価格構造1R診断": d0.get("1R診断") if d0 is not None else "—",
            "採用購入株数": p0.get("採用購入株数") if p0 is not None else np.nan,
            f"Stop時想定損失_{currency}": p0.get(f"Stop時推定総損失_{currency}") if p0 is not None else np.nan,
            "選択Stopの20日結果": sel.get("結果") if sel is not None else "—",
            "選択Stopの20日Net_R": sel.get("Net_R") if sel is not None else np.nan,
            "20日で最良のStop方式": best.get("Stop方式") if best is not None else "—",
            "20日で最良のNet_R": best.get("Net_R") if best is not None else np.nan,
        })
    return pd.DataFrame(rows)


def loss_conversion_rate():
    mode = st.radio("想定損失の円・ドル換算", ["為替を自動取得", "手入力"], horizontal=True)
    if mode == "手入力":
        return st.number_input("1 USD = 何 JPY", min_value=0.0, value=0.0,
            step=0.1, format="%.4f", key="loss_display_usd_jpy")
    if st.button("為替レートを再取得", key="refresh_loss_fx"):
        cached_usd_jpy.clear()
    with st.spinner("ドル円レートを取得しています…"):
        quote = cached_usd_jpy()
    if quote.get("error"):
        st.warning(quote["error"] + " 「手入力」に切り替えて入力できます。")
        return 0.0
    st.metric("自動取得したドル円", f"1 USD = {quote['rate']:.4f} JPY")
    st.caption(f"配信元のデータ時刻：{quote['asof']}。最新の取得可能な1時間足の終値です。"
               "休場中は直近の値を使用します。5分間キャッシュし、再取得ボタンでも更新できます。")
    return quote["rate"]


def show_beginner_cards(result, selected_multiplier, currency):
    summary = beginner_signal_summary(result, selected_multiplier, currency)
    outcomes = result.get("outcomes", pd.DataFrame())
    o20 = outcomes[outcomes["評価期間"] == "20営業日"] if outcomes is not None and not outcomes.empty else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("分析対象", str(result["ticker"]))
    col2.metric("Day0", str(result["day0"].date()))
    col3.metric("20営業日の比較組み合わせ数", f"{len(o20)} 組",
        help="シグナル × Stop方式の組み合わせ数です。取引回数やローソク足の本数ではありません。")
    best_all = o20["Net_R"].max() if not o20.empty else np.nan
    col4.metric("20営業日の最良Net R", fmt_num(best_all, 2, " R"))

    if not o20.empty:
        signal_count = o20["シグナル"].nunique()
        method_count = o20["Stop方式"].nunique()
        st.caption(f"比較対象：シグナル{signal_count}種類・Stop方式{method_count}種類、合計{len(o20)}組。取引回数ではありません。")
        with st.expander("比較している組み合わせを見る"):
            st.dataframe(o20[["シグナル", "Stop方式"]], use_container_width=True, hide_index=True)

    usd_jpy = loss_conversion_rate()
    st.caption("このレートは現在の円・ドル併記に使用します。過去の基準日の為替ではありません。株数・Stop計算の換算レートは『3. 資金管理』で指定します。")
    d0 = first_row(result.get("day0_summary", pd.DataFrame()))
    if d0 is not None:
        st.markdown("#### 基準日の価格とBB状態")
        a, b, c = st.columns(3)
        a.metric("Day0終値", fmt_num(d0.get("Close")))
        b.metric("Day0高値", fmt_num(d0.get("High")))
        c.metric("Day0安値", fmt_num(d0.get("Low")))
        st.write(f"BB状態：**{d0.get('BB状態', '—')}**")
    st.caption("シグナル未成立でも、下の価格チャート・高値安値の波・判定状況を確認できます。")

    st.markdown("#### まず確認する結論（シグナル別）")
    for _, row in summary.iterrows():
        signal = row["シグナル"]
        loss_value = row.get(f"Stop時想定損失_{currency}")
        stop_loss = fmt_num(loss_value, 0, f" {currency}")
        if pd.notna(loss_value) and usd_jpy > 0:
            if str(currency).strip().upper() == "JPY":
                stop_loss += " ／ " + fmt_num(float(loss_value) / usd_jpy, 2, " USD")
            elif str(currency).strip().upper() == "USD":
                stop_loss += " ／ " + fmt_num(float(loss_value) * usd_jpy, 0, " JPY")
        elif str(currency).strip().upper() in ("JPY", "USD") and pd.notna(loss_value):
            stop_loss += "（円・ドル併記には上の換算レートを入力）"
        qty = "—" if pd.isna(row.get("採用購入株数")) else f"{int(row.get('採用購入株数'))} 株"
        best_stop = row.get("20日で最良のStop方式", "—")
        best_r = fmt_num(row.get("20日で最良のNet_R"), 2, " R")
        selected_r = fmt_num(row.get("選択Stopの20日Net_R"), 2, " R")
        diag = row.get("価格構造1R診断", "—")
        status_text = ""
        if pd.notna(row.get("価格構造1R_ATR倍率")):
            atr_ratio = float(row.get("価格構造1R_ATR倍率"))
            if atr_ratio < 0.25:
                status_text = "⚠️ かなり狭いStopです"
            elif atr_ratio < 0.50:
                status_text = "⚠️ 狭めのStopです"
            elif atr_ratio < 1.00:
                status_text = "🟡 ATRよりやや狭いStopです"
            else:
                status_text = "🟢 ATR以上の幅があります"
        with st.container(border=True):
            st.markdown(f"**{signal}**")
            st.write(f"判定状況：**{row['判定状況']}**")
            st.write(row["確認事項"])
            st.write(f"- 価格構造1R診断: **{diag}**  {status_text}")
            st.write(f"- 資金管理で採用される購入株数: **{qty}**")
            st.write(f"- そのStopに到達した場合の想定損失: **{stop_loss}**")
            st.write(f"- 選択中ATR Stop（ATR×{float(selected_multiplier):g}）の20営業日Net R: **{selected_r}**")
            st.write(f"- 20営業日で最も良かったStop方式: **{best_stop}** / **{best_r}**")

    st.dataframe(summary.round(4), use_container_width=True, hide_index=True)


def wave_figure(prices, swings, title, show_bb=True):
    dates = pd.to_datetime(prices["日付"]).dt.strftime("%Y-%m-%d")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.76, 0.24],
        subplot_titles=("株価・高値安値の波", "出来高"),
    )
    fig.add_trace(go.Candlestick(x=dates, open=prices.Open, high=prices.High,
        low=prices.Low, close=prices.Close, name="ローソク足",
        increasing_line_color="#ef5350", increasing_fillcolor="#ef5350",
        decreasing_line_color="#26a69a", decreasing_fillcolor="#26a69a"), row=1, col=1)
    if show_bb:
        for col, name, color in [("BB_Lower", "BB下限", "#42a5f5"),
                                 ("BB_Middle", "BB中央", "#616161"),
                                 ("BB_Upper", "BB上限", "#42a5f5")]:
            fig.add_trace(go.Scatter(x=dates, y=prices[col], name=name,
                mode="lines", line=dict(color=color, width=1)), row=1, col=1)
    for col, name, color in [("MA50", "50日線（SMA）", "#7b1fa2"),
                              ("MA200", "200日線（SMA）", "#e65100")]:
        if col in prices.columns:
            fig.add_trace(go.Scatter(x=dates, y=prices[col], name=name,
                mode="lines", line=dict(color=color, width=2), connectgaps=False), row=1, col=1)
    if not swings.empty:
        sx = pd.to_datetime(swings["日付"]).dt.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(x=sx, y=swings["価格"], mode="lines",
            name="高値・安値の波", line=dict(color="#b8860b", width=2.5)), row=1, col=1)
        for kind, color, symbol in [("高値", "#ef5350", "triangle-down"),
                                     ("安値", "#26a69a", "triangle-up")]:
            q = swings[swings["種類"] == kind]
            fig.add_trace(go.Scatter(x=pd.to_datetime(q["日付"]).dt.strftime("%Y-%m-%d"),
                y=q["価格"], mode="markers+text", name=kind,
                marker=dict(color=color, size=11, symbol=symbol),
                text=[f"{kind} {v:,.2f}" for v in q["価格"]],
                textposition="top center" if kind == "高値" else "bottom center",
                customdata=q[["確定日", "前回同種比"]].astype(str).values,
                hovertemplate="%{x}<br>%{y:,.2f}<br>確定日: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"), row=1, col=1)

    volume = (pd.to_numeric(prices["Volume"], errors="coerce").fillna(0)
              if "Volume" in prices.columns else pd.Series(0, index=prices.index, dtype="float64"))
    candle_colors = np.where(prices["Close"] >= prices["Open"], "#ef5350", "#26a69a")
    fig.add_trace(go.Bar(x=dates, y=volume, name="出来高", marker_color=candle_colors,
        hovertemplate="%{x}<br>出来高: %{y:,.0f}<extra></extra>"), row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", height=790,
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"),
        dragmode="pan", margin=dict(l=15, r=80, t=90, b=45),
        legend=dict(orientation="h", y=1.08), barmode="overlay")
    for axis in ["xaxis", "xaxis2"]:
        fig.layout[axis].update(type="category", categoryorder="array", categoryarray=list(dates),
            rangeslider=dict(visible=False), nticks=10, showspikes=True)
    fig.update_yaxes(side="right", title="価格", showspikes=True, fixedrange=False,
        gridcolor="#e5e7eb", zerolinecolor="#d1d5db", row=1, col=1)
    fig.update_yaxes(side="right", title="出来高（株）", showspikes=True, fixedrange=False,
        gridcolor="#e5e7eb", zerolinecolor="#d1d5db", row=2, col=1)
    for annotation in fig.layout.annotations:
        annotation.font = dict(color="black")
    return fig


def show_full_period_waves(result):
    st.markdown("#### 全期間：高値・安値の波（基準日前後）")
    st.caption("分析開始日から分析終了日までのローソク足です。黒い破線が基準日Day0です。赤は陽線、緑は陰線。紫は50日線、オレンジは200日線、金色は高値と安値を結ぶ波で、その直下に出来高を表示します。")
    width = st.slider("転換点の前後に確認する営業日数", 1, 10, 3,
        help="3なら前後3本より高い高値・低い安値を検出。大きくすると大きな波を見ます。")
    show_bb = st.checkbox("BBバンドを重ねる", value=True)
    path = result["path"].copy()
    path["日付"] = pd.to_datetime(path["日付"])
    if path.empty:
        st.info("指定期間の価格データがありません。日付を確認して再分析してください。")
        return
    swings = confirmed_full_period_swings(path, width)
    fig = wave_figure(path, swings,
        f"{result['ticker']}｜全期間の波・Day0 {result['day0'].date()}", show_bb)
    dates = path["日付"].dt.strftime("%Y-%m-%d").tolist()
    day0_text = pd.Timestamp(result["day0"]).strftime("%Y-%m-%d")
    if day0_text in dates:
        pos = dates.index(day0_text)
        fig.add_shape(type="line", xref="x", yref="paper", x0=pos, x1=pos,
            y0=0, y1=1, line=dict(color="black", width=2, dash="dash"))
        fig.add_annotation(x=pos, y=1, xref="x", yref="paper",
            text=f"Day0 {day0_text}", showarrow=False, yshift=15,
            font=dict(color="black"), bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, theme=None,
        config={"scrollZoom": True, "displaylogo": False})
    st.caption("全期間の事後確認用です。基準日前の転換点でも、確定に基準日以降の足を使う場合があります。確定日は一覧で確認できます。")
    st.caption(f"左右{width}本で確認できた転換点のみ表示します。末尾{width}本は未確定です。"
               "同種の転換点が続く場合はより極端な点を採用し、同日に高値・安値の両方となる足は順序不明のため除外します。"
               "波は表示用で、Stop計算や売買条件には使用しません。")
    if swings.empty:
        st.info("この条件では確定した転換点がありません。確認日数を小さくするか分析期間を広げてください。")
    else:
        with st.expander("高値・安値の一覧と確定日"):
            st.dataframe(swings, use_container_width=True, hide_index=True)
            st.download_button("波の一覧CSVを保存", swings.to_csv(index=False).encode("utf-8-sig"),
                file_name="full_period_swings.csv", mime="text/csv")


def show_readable_comparison(data, value_col, axis_title, key):
    """Full-width horizontal comparison; missing results stay missing."""
    x = data[["方式", value_col]].copy()
    x[value_col] = pd.to_numeric(x[value_col], errors="coerce")
    valid = x[x[value_col].notna() & np.isfinite(x[value_col])].copy()
    if valid.empty:
        st.info("比較できる数値がありません。下の一覧でデータ不足などを確認してください。")
    else:
        labels = valid["方式"].astype(str).str.replace("｜", "<br>", regex=False)
        values = valid[value_col]
        lo, hi = min(0.0, float(values.min())), max(0.0, float(values.max()))
        span = max(hi - lo, 0.1)
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
            marker_color=["#c62828" if v < 0 else "#1565c0" for v in values],
            text=[f"{v:,.3f}" for v in values], textposition="outside",
            textfont=dict(color="black", size=15), cliponaxis=False,
            customdata=valid["方式"],
            hovertemplate="%{customdata}<br>%{x:,.3f}<extra></extra>"))
        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="black", size=15), height=max(380, 72 * len(valid) + 100),
            margin=dict(l=155, r=65, t=25, b=60), showlegend=False,
            xaxis=dict(title=axis_title, range=[lo - span * .22, hi + span * .22],
                       gridcolor="#e5e7eb", zeroline=True, zerolinecolor="black", zerolinewidth=2),
            yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=14), fixedrange=True))
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key,
            config={"displaylogo": False})
    st.dataframe(x.rename(columns={value_col: axis_title}), use_container_width=True,
                 hide_index=True)
    missing = len(x) - len(valid)
    if missing:
        st.caption(f"数値が未確定・計算不可の{missing}組は棒を表示していません。一覧では空欄です。")


def show_single_case_graphics(result):
    path = result.get("path", pd.DataFrame())
    visual = result.get("risk_visual", pd.DataFrame())
    outcomes = result.get("outcomes", pd.DataFrame())

    show_full_period_waves(result)

    st.markdown("#### グラフ1：基準日前後の価格チャート（事後確認用）")
    st.caption("ここでは『Day0以降に反発したか』『BB下限の近くからどう動いたか』を見ます。")
    if path is not None and not path.empty:
        fig = wave_figure(path, pd.DataFrame(),
            f"{result['ticker']}｜基準日前後のローソク足とBBバンド")
        dates = pd.to_datetime(path["日付"]).dt.strftime("%Y-%m-%d").tolist()
        day0_text = pd.Timestamp(result["day0"]).strftime("%Y-%m-%d")
        if day0_text in dates:
            # Category axes use numeric category positions for the reference line.
            day0_pos = dates.index(day0_text)
            fig.add_shape(type="line", xref="x", yref="paper",
                x0=day0_pos, x1=day0_pos, y0=0, y1=1,
                line=dict(color="black", width=2, dash="dash"))
            fig.add_annotation(x=day0_pos, y=1, xref="x", yref="paper",
                text=f"Day0 {day0_text}", showarrow=False,
                yshift=15, font=dict(color="black"), bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, theme=None,
            config={"scrollZoom": True, "displaylogo": False})
        st.caption("黒の破線が基準日Day0です。赤は陽線、緑は陰線。紫は50日線、オレンジは200日線。ドラッグで移動、スクロールで拡大・縮小できます。")
    else:
        st.info("価格チャート用のデータがありません。")

    st.markdown("#### グラフ2：Stop幅（1R%）比較")
    st.caption("右に長い棒ほど、EntryからStopまでの幅が広いです。単位は%です。")
    if visual is not None and not visual.empty:
        show_readable_comparison(visual, "1R_%", "Stop幅（%）", "stop_width_chart")
    else:
        st.info("シグナル未成立またはStop設計不可のため、Stop幅を比較できません。")

    st.markdown("#### グラフ3：1RのATR倍率比較")
    st.caption("1.0ならATR1本分、1.5ならATR×1.5、2.0ならATR×2の幅です。")
    if visual is not None and not visual.empty and "1R_ATR倍率" in visual.columns:
        show_readable_comparison(visual, "1R_ATR倍率", "1RのATR倍率（倍）", "atr_ratio_chart")
    else:
        st.info("ATR倍率を比較できるデータがありません。ダッシュボードの判定状況を確認してください。")

    st.markdown("#### グラフ4：20営業日の実際のNet R比較")
    st.caption("0より右はプラス、左はマイナス。数値は売買コストを差し引いた結果です。")
    if outcomes is not None and not outcomes.empty:
        o20 = outcomes[outcomes["評価期間"] == "20営業日"].copy()
        if not o20.empty:
            o20["方式"] = o20["シグナル"].astype(str) + "｜" + o20["Stop方式"].astype(str)
            show_readable_comparison(o20, "Net_R", "Net R（R）", "net_r_chart")
            with st.expander("未確定・計算不可を含む結果の理由"):
                st.dataframe(o20[["方式", "結果", "決済方法", "Net_R"]],
                             use_container_width=True, hide_index=True)
        else:
            st.info("20営業日の結果データがありません。")
    else:
        st.info("シグナル未成立またはStop設計不可のため、結果を比較できません。")


def show_ledger_graphics(ledger_df):
    st.subheader("13. 蓄積ケースのグラフ比較")
    st.caption("複数ケースを追加した後は、ここで『どのStop方式に傾向があるか』をざっくり確認します。")

    case_list = case_ledger_case_list(ledger_df)
    summary20 = case_ledger_summary(ledger_df, "20営業日")
    all20 = case_ledger_all_ticker_summary(ledger_df, "20営業日")
    comp20 = case_ledger_computability_summary(ledger_df, "20営業日")
    risk_dist = case_ledger_structure_risk_distribution(ledger_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("比較ケース数", f"{ledger_df['Case_ID'].nunique()}")
    m2.metric("比較銘柄数", f"{ledger_df['銘柄'].nunique()}")
    net_count = int(summary20.get("NetR計算可能", pd.Series(dtype=float)).sum()) if not summary20.empty else 0
    m3.metric("20営業日 NetR計算可能件数", f"{net_count}")
    avg_r = all20["Net平均R"].mean() if not all20.empty and "Net平均R" in all20.columns else np.nan
    m4.metric("全体の平均Net R（参考）", fmt_num(avg_r, 2, " R"))

    st.markdown("#### グラフ5：銘柄ごとの蓄積ケース数")
    if case_list is not None and not case_list.empty:
        counts = case_list.groupby("銘柄").size().reset_index(name="ケース数")
        counts["方式"] = counts["銘柄"]
        show_readable_comparison(counts, "ケース数", "ケース数（件）", "ledger_count_chart")
    else:
        st.info("蓄積ケース一覧がありません。")

    st.markdown("#### グラフ6：20営業日 全銘柄参考集計（Net平均R）")
    if all20 is not None and not all20.empty:
        x = all20.copy()
        x["方式"] = x["シグナル"].astype(str) + "｜" + x["Stop方式"].astype(str)
        show_readable_comparison(x, "Net平均R", "Net平均R（R）", "ledger_all_chart")
    else:
        st.info("全銘柄参考集計がありません。")

    st.markdown("#### グラフ7：銘柄別 20営業日 Net平均R")
    if summary20 is not None and not summary20.empty:
        x = summary20.copy()
        x["方式"] = x["銘柄"].astype(str) + "｜" + x["シグナル"].astype(str) + "｜" + x["Stop方式"].astype(str)
        show_readable_comparison(x, "Net平均R", "Net平均R（R）", "ledger_ticker_chart")
    else:
        st.info("銘柄別比較データがありません。")

    st.markdown("#### グラフ8：Net R計算可能率（20営業日）")
    if comp20 is not None and not comp20.empty:
        x = comp20.copy()
        x["方式"] = x["銘柄"].astype(str) + "｜" + x["シグナル"].astype(str) + "｜" + x["Stop方式"].astype(str)
        show_readable_comparison(x, "NetR計算可能率_%", "Net R計算可能率（%）", "ledger_computable_chart")
    else:
        st.info("計算可能率データがありません。")

    st.markdown("#### グラフ9：価格構造1RのATR比帯の分布")
    st.caption("銘柄・シグナルごとに、各ATR比帯の件数を比較します。")
    if risk_dist is not None and not risk_dist.empty:
        bands = [("<0.25ATR件数", "0.25ATR未満", "#c62828"),
                 ("0.25～<0.50ATR件数", "0.25～0.50ATR未満", "#e65100"),
                 ("0.50～<1.00ATR件数", "0.50～1.00ATR未満", "#1565c0"),
                 (">=1.00ATR件数", "1.00ATR以上", "#6a1b9a")]
        x = risk_dist.copy()
        x["対象"] = x["銘柄"].astype(str) + "｜" + x["シグナル"].astype(str)
        existing = [(col, label, color) for col, label, color in bands if col in x]
        if existing:
            fig = go.Figure()
            ymax = 0.0
            for col, label, color in existing:
                values = pd.to_numeric(x[col], errors="coerce")
                ymax = max(ymax, float(values.max()) if values.notna().any() else 0.0)
                fig.add_trace(go.Bar(x=values,
                    y=x["対象"].str.replace("｜", "<br>", regex=False),
                    orientation="h", name=label, marker_color=color,
                    text=[f"{v:.0f}" if pd.notna(v) else "" for v in values],
                    textposition="outside", cliponaxis=False,
                    textfont=dict(color="black", size=14), customdata=x["対象"],
                    hovertemplate="%{customdata}<br>%{x}件<extra>%{fullData.name}</extra>"))
            fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                font=dict(color="black", size=14), barmode="group",
                height=max(480, len(x) * 160 + 180), margin=dict(l=155, r=70, t=155, b=60),
                legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
                xaxis=dict(title="件数（件）", range=[0, max(ymax, 1) * 1.25], dtick=1 if ymax < 10 else None,
                           gridcolor="#e5e7eb"),
                yaxis=dict(autorange="reversed", automargin=True, fixedrange=True))
            st.plotly_chart(fig, use_container_width=True, theme=None,
                key="ledger_atr_distribution_chart", config={"displaylogo": False})
            st.dataframe(x[["対象"] + [col for col, _, _ in existing]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("ATR比帯の分布列がありません。")
    else:
        st.info("ATR比帯分布データがありません。")


if run:
    if quote_to_capital_fx is None or quote_to_capital_fx<=0:
        st.error("通貨換算レートがありません。自動レートを再取得するか、手入力へ切り替えてください。")
        st.stop()
    if analysis_start > case_date:
        st.error("分析開始日はBB基準日以前にしてください。")
        st.stop()
    if analysis_end < case_date:
        st.error("分析終了日はBB基準日以降にしてください。")
        st.stop()
    if symbol_budget > total_capital and total_capital > 0:
        st.warning("1銘柄予算が総資金を上回っています。株数計算では入力された1銘柄予算を使用します。")
    multipliers = parse_multipliers(atr_multipliers_text)
    with st.spinner("株価データを取得し、前後期間・ATR・資金管理まで計算しています..."):
        result = run_case_study(
            ticker, case_date, analysis_start, analysis_end,
            commission_pct / 100, slippage_pct / 100,
            int(atr_period), multipliers, float(selected_atr_multiplier),
            float(total_capital), float(symbol_budget), float(risk_pct),
            float(quote_to_capital_fx), capital_currency.strip() or "資金通貨"
        )
    if result.get("error"):
        st.error(result["error"])
        st.stop()
    st.session_state.current_result = result
    st.session_state.current_settings = {"commission": commission_pct / 100, "slippage": slippage_pct / 100}

result = st.session_state.current_result
if result:
    st.success(f"{result['ticker']} ｜ Day0 {result['day0'].date()} ｜ 分析期間 {result['analysis_start'].date()} ～ {result['analysis_end'].date()}")
    if result["date_note"] != "入力日を使用":
        st.warning(result["date_note"])

    st.subheader("5. まず最初に見るダッシュボード")
    st.caption("このセクションだけで、何を確認すべきかが分かるようにしています。")
    show_beginner_cards(result, selected_atr_multiplier, capital_currency)
    show_single_case_graphics(result)
    show_market_context(result)
    show_ai_panel(result)

    st.divider()
    st.subheader("6. 詳細データ（従来表示）")
    section(1, "ケース日・BB下限位置・ATR・市場状態", result["day0_summary"], True)
    section(2, "基準日前の環境サマリー", result["pre_summary"], False)
    section(3, "4営業日シグナル監査", result["signal_window"], True)
    section(4, "Stop・1R比較", result["risk_design"], True)
    diag = result["risk_diagnostic"]
    if diag is not None and not diag.empty:
        bad = diag[diag["1R診断"].isin(["極端に狭い（価格構造1R<0.25ATR）", "狭い（価格構造1R<0.50ATR）"])]
        if not bad.empty:
            st.warning("価格構造Stopの1RがATRに対して狭いケースがあります。グラフ2・3で確認してください。")
    section(5, "価格構造1R・ATR比診断", diag, True)
    section(6, "予算・許容損失から購入株数を計算", result["position_sizing"], True)
    section(7, "購入後のStop・Target金額損益", result["money_scenarios"], True)
    section(8, "Stop方式別・実際の結果", result["outcomes"], True)
    section(9, "設定期間の価格経路", result["path"], False)
    section(10, "1R幅・ATR換算比較", result["risk_visual"], False)

    st.divider()
    st.subheader("7. 今回のケースを比較台帳へ追加")
    st.write("このボタンでは、今回の分析結果を比較用CSV台帳へ追加するだけです。同じ銘柄・同じDay0を再追加した場合は最新結果で置き換えます。")
    if st.button("今回のケースを比較台帳に追加", type="primary", use_container_width=True):
        settings = st.session_state.current_settings or {"commission": 0.001, "slippage": 0.001}
        new_rows = build_case_ledger_rows(result, settings["commission"], settings["slippage"])
        st.session_state.case_ledger = merge_case_ledgers(st.session_state.case_ledger, new_rows)
        st.success(f"{result['ticker']} / Day0 {result['day0'].date()} を追加しました。現在 {st.session_state.case_ledger['Case_ID'].nunique()} ケースです。")

st.divider()
st.subheader("8. 複数銘柄・複数BB下限ケース比較")
ledger = normalize_case_ledger(st.session_state.case_ledger)
if ledger.empty:
    st.info("まだ比較台帳にケースがありません。1ケースを分析して『今回のケースを比較台帳に追加』を押してください。")
else:
    st.download_button(
        "最新の比較台帳CSVを保存",
        data=ledger.to_csv(index=False, float_format="%.6f").encode("utf-8-sig"),
        file_name="bb_case_ledger_v2_4.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_bottom",
    )

    show_ledger_graphics(ledger)

    section(11, "蓄積ケース一覧", case_ledger_case_list(ledger), True)
    section(12, "20営業日・銘柄別Stop比較", case_ledger_summary(ledger, "20営業日"), True)
    section(13, "20営業日・全銘柄参考集計", case_ledger_all_ticker_summary(ledger, "20営業日"), False)
    section(14, "蓄積ケース明細", case_ledger_detail(ledger), False)
    section(15, "20営業日・Net R計算可能率監査", case_ledger_computability_summary(ledger, "20営業日"), True)
    section(16, "20営業日・銘柄等重み参考集計", case_ledger_equal_ticker_summary(ledger, "20営業日"), True)
    section(17, "20営業日・価格構造Stopとの差（同一ケース）", case_ledger_paired_vs_structure(ledger, "20営業日"), True)
    section(18, "価格構造1R・ATR倍率分布", case_ledger_structure_risk_distribution(ledger), True)
    section(19, "価格構造1R・ATR比ケース明細", case_ledger_structure_risk_cases(ledger), False)

    st.info("【読み方】まず『グラフ6』『グラフ7』でStop方式の傾向をざっくり見て、その後に12～19の表で理由を確認してください。")
    st.warning("平均Rが高いStop方式を、そのまま採用してはいけません。ケース数、銘柄偏り、未決着、データ不足も一緒に確認してください。")

st.warning("これは過去ケースの研究・資金管理シミュレーションです。売買推奨ではありません。価格構造1R診断は警告表示であり、自動的なStop変更・除外条件ではありません。")
