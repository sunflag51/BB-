from datetime import date, timedelta
from io import StringIO
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from case_study_core import (
    APP_VERSION, confirmed_pre_day0_swings, run_case_study, empty_case_ledger, build_case_ledger_rows,
    merge_case_ledgers, normalize_case_ledger, case_ledger_case_list,
    case_ledger_summary, case_ledger_all_ticker_summary,
    case_ledger_computability_summary, case_ledger_equal_ticker_summary,
    case_ledger_paired_vs_structure, case_ledger_structure_risk_distribution,
    case_ledger_structure_risk_cases, case_ledger_detail,
)

st.set_page_config(page_title="自由銘柄・自由期間 BB下限ケース分析", page_icon="🔎", layout="wide")
st.title("🔎 自由銘柄・自由期間 BB下限ケース分析")
st.caption(f"Version {APP_VERSION} ｜ 1ケース分析を維持＋初心者向けグラフィック表示を追加")
st.info("このアプリは『何を見ればよいか分からない』を減らすため、表だけでなく、先に見るべきポイントをカード・グラフで表示します。AI売買判定は行いません。")

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
st.subheader("1. 銘柄と分析期間")
c1, c2 = st.columns(2)
with c1:
    ticker = st.text_input("銘柄コード", value="COST", help="例: COST / AAPL / NVDA / 6857.T")
with c2:
    case_date = st.date_input("BB下限付近の基準日（Day0）", value=date.today() - timedelta(days=30))
c3, c4 = st.columns(2)
with c3:
    analysis_start = st.date_input("基準日前の分析開始日", value=date.today() - timedelta(days=120))
with c4:
    analysis_end = st.date_input("分析終了日", value=date.today())

st.subheader("2. 1R・Stopの比較条件")
c5, c6, c7 = st.columns(3)
with c5:
    atr_period = st.number_input("ATR期間", 5, 100, 14, 1, help="ATR = Average True Range。直近の値幅の大きさです。")
with c6:
    atr_multipliers_text = st.text_input("比較するATR倍率", value="1.0,1.5,2.0", help="カンマ区切り。例: 1.0,1.5,2.0,2.5")
with c7:
    selected_atr_multiplier = st.number_input("資金管理で主に使うATR倍率", 0.1, 10.0, 1.5, 0.1, format="%.1f")

st.subheader("3. 資金管理")
st.caption("米国株を円資金で見る場合は、換算レートに『1 USD = 何円』を入力してください。日本株を円で見る場合は1.0です。")
c8, c9, c10 = st.columns(3)
with c8:
    total_capital = st.number_input("総資金", min_value=0.0, value=1000000.0, step=10000.0, format="%.2f")
with c9:
    symbol_budget = st.number_input("この1銘柄に使える予算", min_value=0.0, value=300000.0, step=10000.0, format="%.2f")
with c10:
    risk_pct = st.number_input("1取引の許容損失（総資金に対する%）", 0.01, 100.0, 1.0, 0.1, format="%.2f")
c11, c12 = st.columns(2)
with c11:
    quote_to_capital_fx = st.number_input("株価通貨→資金通貨 換算レート", min_value=0.000001, value=1.0, step=0.1, format="%.4f")
with c12:
    capital_currency = st.text_input("資金通貨の表示名", value="JPY", help="例: JPY / USD")

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
        if d0 is None and p0 is None and best is None and sel is None:
            continue
        rows.append({
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


def show_beginner_cards(result, selected_multiplier, currency):
    summary = beginner_signal_summary(result, selected_multiplier, currency)
    outcomes = result.get("outcomes", pd.DataFrame())
    o20 = outcomes[outcomes["評価期間"] == "20営業日"] if outcomes is not None and not outcomes.empty else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("分析対象", str(result["ticker"]))
    col2.metric("Day0", str(result["day0"].date()))
    col3.metric("20営業日の比較本数", f"{len(o20)}")
    best_all = o20["Net_R"].max() if not o20.empty else np.nan
    col4.metric("20営業日の最良Net R", fmt_num(best_all, 2, " R"))

    if summary.empty:
        st.info("初心者向けサマリーを作れるデータがありません。")
        return

    st.markdown("#### まず確認する結論（シグナル別）")
    for _, row in summary.iterrows():
        signal = row["シグナル"]
        stop_loss = fmt_num(row.get(f"Stop時想定損失_{currency}"), 0, f" {currency}")
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
            st.write(f"- 価格構造1R診断: **{diag}**  {status_text}")
            st.write(f"- 資金管理で採用される購入株数: **{qty}**")
            st.write(f"- そのStopに到達した場合の想定損失: **{stop_loss}**")
            st.write(f"- 選択中ATR Stop（ATR×{float(selected_multiplier):g}）の20営業日Net R: **{selected_r}**")
            st.write(f"- 20営業日で最も良かったStop方式: **{best_stop}** / **{best_r}**")

    st.dataframe(summary.round(4), use_container_width=True, hide_index=True)


def wave_figure(prices, swings, title, show_bb=True):
    fig = go.Figure()
    dates = pd.to_datetime(prices["日付"]).dt.strftime("%Y-%m-%d")
    fig.add_trace(go.Candlestick(x=dates, open=prices.Open, high=prices.High,
        low=prices.Low, close=prices.Close, name="ローソク足",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    if show_bb:
        for col, name, color in [("BB_Lower", "BB下限", "#42a5f5"),
                                 ("BB_Middle", "BB中央", "#a0a7b4"),
                                 ("BB_Upper", "BB上限", "#42a5f5")]:
            fig.add_trace(go.Scatter(x=dates, y=prices[col], name=name,
                mode="lines", line=dict(color=color, width=1)))
    if not swings.empty:
        sx = pd.to_datetime(swings["日付"]).dt.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(x=sx, y=swings["価格"], mode="lines",
            name="高値・安値の波", line=dict(color="#ffc107", width=2.5)))
        for kind, color, symbol in [("高値", "#ef5350", "triangle-down"),
                                     ("安値", "#26a69a", "triangle-up")]:
            q = swings[swings["種類"] == kind]
            fig.add_trace(go.Scatter(x=pd.to_datetime(q["日付"]).dt.strftime("%Y-%m-%d"),
                y=q["価格"], mode="markers+text", name=kind,
                marker=dict(color=color, size=11, symbol=symbol),
                text=[f"{kind} {v:,.2f}" for v in q["価格"]],
                textposition="top center" if kind == "高値" else "bottom center",
                customdata=q[["確定日", "前回同種比"]].astype(str).values,
                hovertemplate="%{x}<br>%{y:,.2f}<br>確定日: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(title=title, template="plotly_dark", height=600,
        paper_bgcolor="#131722", plot_bgcolor="#131722", dragmode="pan",
        margin=dict(l=15, r=70, t=65, b=45),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(dates),
                   rangeslider=dict(visible=False), nticks=10, showspikes=True),
        yaxis=dict(side="right", title="価格", showspikes=True, fixedrange=False))
    return fig


def show_pre_day0_waves(result):
    st.markdown("#### 基準日前：高値・安値の波")
    st.caption("Day0当日とそれ以降を除いたローソク足です。黄色の線で高値と安値の波を結びます。")
    width = st.slider("転換点の前後に確認する営業日数", 1, 10, 3,
        help="3なら前後3本より高い高値・低い安値を検出。大きくすると大きな波を見ます。")
    show_bb = st.checkbox("BBバンドを重ねる", value=True)
    path = result["path"].copy()
    path["日付"] = pd.to_datetime(path["日付"])
    pre = path[path["日付"] < pd.Timestamp(result["day0"])].copy()
    if pre.empty:
        st.info("基準日前のデータがありません。分析開始日を早めて再分析してください。")
        return
    swings = confirmed_pre_day0_swings(path, result["day0"], width)
    st.plotly_chart(wave_figure(pre, swings,
        f"{result['ticker']}｜Day0 {result['day0'].date()} より前", show_bb),
        use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    st.caption(f"左右{width}本で確認できた転換点のみ表示します。末尾{width}本は未確定です。"
               "同種の転換点が続く場合はより極端な点を採用し、同日に高値・安値の両方となる足は順序不明のため除外します。"
               "波は表示用で、Stop計算や売買条件には使用しません。")
    if swings.empty:
        st.info("この条件では確定した転換点がありません。確認日数を小さくするか分析期間を広げてください。")
    else:
        with st.expander("高値・安値の一覧と確定日"):
            st.dataframe(swings, use_container_width=True, hide_index=True)
            st.download_button("波の一覧CSVを保存", swings.to_csv(index=False).encode("utf-8-sig"),
                file_name="pre_day0_swings.csv", mime="text/csv")


def show_single_case_graphics(result):
    path = result.get("path", pd.DataFrame())
    visual = result.get("risk_visual", pd.DataFrame())
    outcomes = result.get("outcomes", pd.DataFrame())

    show_pre_day0_waves(result)

    st.markdown("#### グラフ1：基準日前後の価格チャート（事後確認用）")
    st.caption("ここでは『Day0以降に反発したか』『BB下限の近くからどう動いたか』を見ます。")
    if path is not None and not path.empty:
        price_chart = path.set_index(pd.to_datetime(path["日付"]))[["Close", "BB_Lower", "BB_Middle", "BB_Upper"]]
        st.line_chart(price_chart, use_container_width=True)
    else:
        st.info("価格チャート用のデータがありません。")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### グラフ2：Stop幅（1R%）比較")
        st.caption("棒が高いほど、EntryからStopまでの幅が広いです。")
        if visual is not None and not visual.empty:
            st.bar_chart(visual.set_index("方式")[["1R_%"]], use_container_width=True)
        else:
            st.info("1R比較データがありません。")
    with c2:
        st.markdown("#### グラフ3：1RのATR倍率比較")
        st.caption("1.0付近ならATR1本分、1.5ならATR×1.5、2.0ならATR×2の幅です。")
        if visual is not None and not visual.empty and "1R_ATR倍率" in visual.columns:
            st.bar_chart(visual.set_index("方式")[["1R_ATR倍率"]], use_container_width=True)
        else:
            st.info("ATR倍率比較データがありません。")

    st.markdown("#### グラフ4：20営業日の実際のNet R比較")
    st.caption("プラスなら0より上、マイナスなら0より下になります。")
    if outcomes is not None and not outcomes.empty:
        o20 = outcomes[outcomes["評価期間"] == "20営業日"].copy()
        if not o20.empty:
            o20["方式"] = o20["シグナル"].astype(str) + "｜" + o20["Stop方式"].astype(str)
            st.bar_chart(o20.set_index("方式")[["Net_R"]], use_container_width=True)
        else:
            st.info("20営業日の結果データがありません。")
    else:
        st.info("実際の結果データがありません。")


def show_ledger_graphics(ledger_df):
    st.subheader("14. 蓄積ケースのグラフ比較")
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

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### グラフ5：銘柄ごとの蓄積ケース数")
        if case_list is not None and not case_list.empty:
            counts = case_list.groupby("銘柄").size().to_frame("ケース数")
            st.bar_chart(counts, use_container_width=True)
        else:
            st.info("蓄積ケース一覧がありません。")
    with g2:
        st.markdown("#### グラフ6：20営業日 全銘柄参考集計（Net平均R）")
        if all20 is not None and not all20.empty:
            all20c = all20.copy()
            all20c["方式"] = all20c["シグナル"].astype(str) + "｜" + all20c["Stop方式"].astype(str)
            st.bar_chart(all20c.set_index("方式")[["Net平均R"]], use_container_width=True)
        else:
            st.info("全銘柄参考集計がありません。")

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("#### グラフ7：銘柄別 20営業日 Net平均R")
        if summary20 is not None and not summary20.empty:
            s = summary20.copy()
            s["方式"] = s["銘柄"].astype(str) + "｜" + s["シグナル"].astype(str) + "｜" + s["Stop方式"].astype(str)
            st.bar_chart(s.set_index("方式")[["Net平均R"]], use_container_width=True)
        else:
            st.info("銘柄別比較データがありません。")
    with g4:
        st.markdown("#### グラフ8：Net R計算可能率（20営業日）")
        if comp20 is not None and not comp20.empty:
            c = comp20.copy()
            c["方式"] = c["銘柄"].astype(str) + "｜" + c["シグナル"].astype(str) + "｜" + c["Stop方式"].astype(str)
            st.bar_chart(c.set_index("方式")[["NetR計算可能率_%"]], use_container_width=True)
        else:
            st.info("計算可能率データがありません。")

    st.markdown("#### グラフ9：価格構造1RのATR比帯の分布")
    if risk_dist is not None and not risk_dist.empty:
        cols = [
            "<0.25ATR件数",
            "0.25～<0.50ATR件数",
            "0.50～<1.00ATR件数",
            ">=1.00ATR件数",
        ]
        existing = [c for c in cols if c in risk_dist.columns]
        if existing:
            rd = risk_dist.copy()
            rd["対象"] = rd["銘柄"].astype(str) + "｜" + rd["シグナル"].astype(str)
            st.bar_chart(rd.set_index("対象")[existing], use_container_width=True)
        else:
            st.info("ATR比帯の分布列がありません。")
    else:
        st.info("ATR比帯分布データがありません。")


if run:
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
