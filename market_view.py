"""US index context and 11-sector rotation; display-only calculations."""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

INDICES = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ総合", "^DJI": "NYダウ", "^RUT": "Russell 2000"}
SECTORS = {
    "XLK": ("情報技術", "#1f77b4"), "XLF": ("金融", "#ff7f0e"),
    "XLV": ("ヘルスケア", "#2ca02c"), "XLY": ("一般消費財", "#d62728"),
    "XLP": ("生活必需品", "#9467bd"), "XLE": ("エネルギー", "#8c564b"),
    "XLI": ("資本財", "#e377c2"), "XLB": ("素材", "#bcbd22"),
    "XLU": ("公益事業", "#17becf"), "XLRE": ("不動産", "#4b0082"),
    "XLC": ("通信サービス", "#008080"),
}

SECTOR_KEYS = {
    "technology": "XLK", "financial-services": "XLF", "healthcare": "XLV",
    "consumer-cyclical": "XLY", "consumer-defensive": "XLP", "energy": "XLE",
    "industrials": "XLI", "basic-materials": "XLB", "utilities": "XLU",
    "real-estate": "XLRE", "communication-services": "XLC",
}
SECTOR_NAMES = {
    "Technology": "XLK", "Financial Services": "XLF", "Financials": "XLF",
    "Healthcare": "XLV", "Health Care": "XLV", "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY", "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Energy": "XLE", "Industrials": "XLI", "Basic Materials": "XLB", "Materials": "XLB",
    "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC",
}

@st.cache_data(ttl=86400, show_spinner=False)
def lookup_stock_sector(ticker):
    ticker = str(ticker).strip().upper()
    if ticker in SECTORS:
        return {"symbol": ticker, "source": "選択されたセクターETF", "raw": SECTORS[ticker][0]}
    try:
        info = yf.Ticker(ticker).get_info()
        symbol = SECTOR_KEYS.get(info.get("sectorKey")) or SECTOR_NAMES.get(info.get("sector"))
        return {"symbol": symbol, "source": "Yahoo Financeの企業分類", "raw": info.get("sector", "分類なし")}
    except Exception:
        return {"symbol": None, "source": "分類の取得失敗", "raw": "分類を自動取得できませんでした"}


def selected_sector_control(ticker):
    with st.spinner("選択銘柄のセクターを確認しています…"):
        classification = lookup_stock_sector(ticker)
    options = [""] + list(SECTORS)
    detected = classification["symbol"] or ""
    selected = st.selectbox("選択銘柄の所属セクター（自動判定結果を変更できます）", options,
        index=options.index(detected) if detected in options else 0,
        format_func=lambda s: "未分類・指定なし" if not s else f"{SECTORS[s][0]} ({s})",
        key=f"chosen_sector_{ticker}")
    st.caption(f"分類元：{classification['source']}／取得分類：{classification['raw']}。現在の企業分類を使用し、過去の分類変更は再現しません。")
    if selected:
        kind = "自動判定" if selected == detected else "手動指定"
        st.success(f"★ {ticker} → {SECTORS[selected][0]} ({selected})｜{kind}")
        st.caption("★は選択銘柄の所属セクターの位置です。銘柄自体の騰落率や、セクター内での銘柄順位ではありません。米国セクターETFを比較対象に使用します。")
    else:
        st.info("セクターを特定できません。上の欄で指定すると、グラフに★を表示します。幅広い指数ETFには単一の所属セクターを割り当てません。")
    return selected or None


@st.cache_data(ttl=600, show_spinner=False)
def download_market_prices(start, end):
    symbols = list(INDICES) + list(SECTORS) + ["SPY"]
    try:
        raw = yf.download(symbols, start=(pd.Timestamp(start)-pd.Timedelta(days=550)).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end)+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=4, timeout=10)
    except Exception:
        return pd.DataFrame(), symbols
    columns = {}
    for symbol in symbols:
        try:
            series = raw[symbol]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            series = pd.to_numeric(series, errors="coerce")
            series = series.where(np.isfinite(series) & series.gt(0))
            series.index = pd.to_datetime(series.index)
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            series.index = series.index.normalize()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            if series.notna().any():
                columns[symbol] = series
        except (KeyError, TypeError, AttributeError):
            continue
    prices = pd.DataFrame(columns).sort_index()
    missing = [symbol for symbol in symbols if symbol not in prices]
    return prices, missing


def trailing_return(prices, horizon):
    # A gap inside the window invalidates the comparison; no fill or interpolation.
    valid = prices.rolling(horizon+1, min_periods=horizon+1).count().eq(horizon+1)
    return (prices/prices.shift(horizon)-1).mul(100).where(valid)


def calculate_market(prices, start, end, horizon):
    if "^GSPC" not in prices or prices["^GSPC"].dropna().empty:
        return {"error": "取引日基準となるS&P 500を取得できませんでした。再取得してください。"}
    calendar = prices.index[prices["^GSPC"].notna()]
    p = prices.reindex(calendar).sort_index()
    display = (p.index >= pd.Timestamp(start)) & (p.index <= pd.Timestamp(end))
    market = p.reindex(columns=list(INDICES))
    returns = trailing_return(market, horizon).loc[display]
    common = market.loc[display].dropna()
    normalized = common.div(common.iloc[0]).mul(100) if not common.empty else common
    # No partial-universe average when any of the four indices is unavailable.
    score = returns.mean(axis=1).where(returns.notna().all(axis=1))
    sector_prices = p.reindex(columns=list(SECTORS))
    sectors = trailing_return(sector_prices, horizon).loc[display]
    spy = trailing_return(p.reindex(columns=["SPY"]), horizon)["SPY"].loc[display]
    relative = sectors.sub(spy, axis=0)
    leaders = []
    for dt, row in sectors.iterrows():
        if not row.notna().all():
            leaders.append({"日付": dt, "ETF": "", "主導セクター": "データ不足", "騰落率_%": np.nan, "SPY超過_pp": np.nan, "状態": "11セクターすべてが揃わないため未判定"})
            continue
        maximum = float(row.max())
        winners = row.index[np.isclose(row.values, maximum, rtol=0, atol=1e-10)]
        if len(winners) != 1:
            leaders.append({"日付": dt, "ETF": "", "主導セクター": "同率首位", "騰落率_%": maximum, "SPY超過_pp": np.nan, "状態": "同率首位: " + " / ".join(winners)})
            continue
        symbol = winners[0]
        leaders.append({"日付": dt, "ETF": symbol, "主導セクター": SECTORS[symbol][0], "騰落率_%": maximum,
            "SPY超過_pp": relative.loc[dt, symbol], "状態": "相対首位・全セクター下落" if maximum < 0 else "騰落率首位"})
    return {"error": None, "normalized": normalized, "market_returns": returns, "score": score,
            "sectors": sectors, "relative": relative, "leaders": pd.DataFrame(leaders)}


def style(fig, height=450):
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black", size=14), height=height, margin=dict(l=40, r=35, t=85, b=60),
        legend=dict(orientation="h", y=1.08), hovermode="x unified")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True, gridcolor="#e5e7eb")
    return fig


def day0_line(fig, dates, day0):
    stamp = pd.Timestamp(day0).strftime("%Y-%m-%d")
    if stamp in dates:
        pos = dates.index(stamp)
        fig.add_shape(type="line", xref="x", yref="paper", x0=pos, x1=pos, y0=0, y1=1,
            line=dict(color="black", width=1.5, dash="dash"))


def show_market_context(result):
    st.subheader("市場の動向・セクターローテーション（米国市場）")
    st.caption("米国4指数と米国11セクターETFで計算します。日本株などを分析中でも、この欄は米国市場の参考表示です。売買・Stopの計算には使用しません。")
    ticker = str(result["ticker"]).upper()
    selected = selected_sector_control(ticker)
    horizon = st.selectbox("市場・セクターの比較期間（営業日）", [5, 20, 60], index=1, key="market_horizon")
    if st.button("市場・セクターデータを再取得", key="refresh_market"):
        download_market_prices.clear()
    start, end = result["analysis_start"], result["analysis_end"]
    with st.spinner("米国指数と11セクターの価格を取得しています…"):
        prices, missing = download_market_prices(start, end)
    if missing:
        st.warning("取得できなかったデータ: " + ", ".join(missing) + "。欠損を0や前日の値で埋めず、計算できる項目だけ表示します。")
    data = calculate_market(prices, start, end, int(horizon))
    if data["error"]:
        st.warning(data["error"])
        return
    with st.expander("対象の指数・セクターと計算方法", expanded=False):
        st.dataframe(pd.DataFrame([{"指数": name, "取得コード": symbol} for symbol, name in INDICES.items()]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame([{"セクター": name, "ETF": symbol, "色": color} for symbol, (name,color) in SECTORS.items()]), hide_index=True, use_container_width=True)
        st.write("指数比較＝同じ基準日の終値を100として、その日の終値÷基準日終値×100。基準日は表示期間内で4指数が揃う最初の日。")
        st.write(f"{horizon}営業日騰落率＝（当日終値÷{horizon}営業日前終値−1）×100。市場スコア＝4指数の騰落率の単純平均。")
        st.write("主導セクター＝11セクターETFの騰落率の首位。同率首位・欠損は灰色。全セクター下落時の首位は『相対首位』です。")
        st.write("セクターは分配金・分割調整後の価格を使用。SPY超過＝セクター騰落率−SPY騰落率（単位pp）。市場指数は価格指数で、ETFと収益定義が異なります。")
        st.write("取引日はS&P 500の観測日を使用。各日の判定はその日以前のデータだけで計算。欠損のある期間は未判定。")

    st.markdown("#### 市場グラフ：4指数を同じ日から100で比較")
    norm = data["normalized"]
    if norm.empty:
        st.info("4指数が同じ日に揃うデータがなく、100基準比較は表示できません。")
    else:
        dates = norm.index.strftime("%Y-%m-%d").tolist()
        fig = go.Figure()
        for symbol, name in INDICES.items():
            fig.add_trace(go.Scatter(x=dates, y=norm[symbol], mode="lines", name=f"★ {ticker} → {name} ({symbol})" if symbol == selected else f"{name} ({symbol})"))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=dates, nticks=10)
        fig.update_yaxes(title="基準日＝100")
        day0_line(fig, dates, result["day0"])
        st.plotly_chart(style(fig), theme=None, use_container_width=True)
        st.caption(f"共通の基準日：{dates[0]}／最終共通データ日：{dates[-1]}。黒い破線はDay0です。")
    returns = data["market_returns"]
    valid_score = data["score"].dropna()
    st.markdown(f"#### 市場スコア：4指数の{horizon}営業日騰落率平均")
    if not valid_score.empty:
        dates = valid_score.index.strftime("%Y-%m-%d").tolist()
        fig = go.Figure(go.Scatter(x=dates, y=valid_score, mode="lines", name="4指数平均騰落率"))
        fig.update_xaxes(type="category", nticks=10)
        fig.update_yaxes(title="平均騰落率（%）", zeroline=True, zerolinecolor="black")
        day0_line(fig, dates, result["day0"])
        st.plotly_chart(style(fig, 320), theme=None, use_container_width=True)
    st.dataframe(returns.rename(columns=INDICES).reset_index(names="日付"), hide_index=True, use_container_width=True)

    sectors = data["sectors"]
    if sectors.empty:
        st.info("指定期間のセクターデータがありません。")
        return
    dates = sectors.index.strftime("%Y-%m-%d").tolist()
    st.markdown(f"#### セクターの強弱：{horizon}営業日騰落率")
    st.caption("赤は下落、緑は上昇。白は0%付近です。空欄は計算に必要なデータ不足です。")
    bound = np.nanmax(np.abs(sectors.to_numpy())) if sectors.notna().any().any() else 1.0
    heat_labels = [f"★ {ticker} → {name} ({symbol})" if symbol == selected else f"{name} ({symbol})"
                   for symbol,(name,_) in SECTORS.items()]
    fig = go.Figure(go.Heatmap(x=dates, y=heat_labels,
        z=sectors.T.to_numpy(), zmin=-max(bound,1), zmax=max(bound,1),
        colorscale=[[0,"#c62828"],[.5,"white"],[1,"#238b45"]], colorbar=dict(title="%"),
        hovertemplate="%{x}<br>%{y}<br>%{z:.2f}%<extra></extra>", hoverongaps=False))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=dates, nticks=10)
    fig.update_yaxes(autorange="reversed")
    day0_line(fig, dates, result["day0"])
    style(fig,600)
    if selected:
        position = list(SECTORS).index(selected)
        fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1,
            y0=position-.48, y1=position+.48, line=dict(color="black", width=3),
            fillcolor="rgba(0,0,0,0)")
        fig.update_layout(margin=dict(l=210, r=35, t=85, b=60))
    st.plotly_chart(fig, theme=None, use_container_width=True)

    st.markdown("#### 主導セクターの推移：横方向のカラー帯")
    leaders = data["leaders"]
    palette = ["#bdbdbd"] + [color for _,color in SECTORS.values()]
    codes = {symbol:i+1 for i,symbol in enumerate(SECTORS)}
    n = len(palette)
    scale = []
    for i,color in enumerate(palette):
        scale.extend([[i/n,color],[(i+1)/n,color]])
    band_rows = ["全11セクターの首位"]
    band_values = [[codes.get(s,0) for s in leaders["ETF"]]]
    leader_details = leaders[["主導セクター", "ETF", "騰落率_%", "SPY超過_pp", "状態"]].to_numpy()
    own_returns = sectors[selected] if selected else pd.Series(np.nan,index=sectors.index)
    ranks = sectors.rank(axis=1,ascending=False,method="min").where(sectors.notna().all(axis=1), axis=0)
    own_rank = ranks[selected] if selected else pd.Series(np.nan,index=sectors.index)
    details = np.column_stack([leader_details,own_returns.to_numpy(),own_rank.to_numpy()])
    if selected:
        band_rows.append(f"★ {ticker}のセクターが首位")
        band_values.append([codes[selected] if symbol == selected else 0 for symbol in leaders["ETF"]])
    custom = np.repeat(details[None,:,:],len(band_rows),axis=0)
    fig = go.Figure(go.Heatmap(x=dates, y=band_rows,
        z=band_values, zmin=-.5,zmax=n-.5,
        colorscale=scale, showscale=False, customdata=custom,
        hovertemplate="%{x}<br>%{y}<br>全体首位: %{customdata[0]} (%{customdata[1]})<br>首位騰落率: %{customdata[2]:.2f}%<br>%{customdata[4]}<br>選択セクター騰落率: %{customdata[5]:.2f}%<br>選択セクター順位: %{customdata[6]:.0f} / 11<extra></extra>"))
    # Separate legend entries keep sector colors identifiable without overcrowding the strip.
    for symbol,(name,color) in SECTORS.items():
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(color=color,size=12),name=f"★ {ticker} → {name} ({symbol})" if symbol == selected else f"{name} ({symbol})"))
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(color=palette[0],size=12),name="未判定・同率首位"))
    style(fig,420 if selected else 360)
    fig.update_layout(hovermode="closest",margin=dict(l=90,r=25,t=25,b=170),
        legend=dict(orientation="h",y=-.5,yanchor="top",x=0,font=dict(size=12)))
    fig.update_xaxes(type="category",categoryorder="array",categoryarray=dates,nticks=10)
    fig.update_yaxes(showgrid=False,autorange="reversed",automargin=True)
    day0_line(fig,dates,result["day0"])
    st.plotly_chart(fig,theme=None,use_container_width=True)
    if selected:
        st.caption(f"下の★行は{ticker}の所属セクターが単独首位の日だけ色を付けます。灰色は首位以外・データ不足・同率首位です。日付にカーソルを置くと、その日の11セクター中の順位も確認できます。")
    st.caption("色はセクターの種類です。日付にカーソルを置くとセクター名・騰落率・SPYとの差を表示します。灰色は11セクター不足または同率首位です。")

    complete = sectors.dropna()
    st.markdown("#### 最新共通日のセクター順位")
    if complete.empty:
        st.info("11セクターすべてを比較できる日がありません。主導セクターは未判定です。")
    else:
        asof = complete.index[-1]
        row = complete.loc[asof].sort_values()
        labels = [f"★ {ticker} → {SECTORS[s][0]} ({s})" if s == selected else f"{SECTORS[s][0]} ({s})" for s in row.index]
        if selected:
            rank = int(complete.loc[asof].rank(ascending=False,method="min")[selected])
            a,b,c = st.columns(3)
            a.metric(f"★ {ticker}の所属セクター順位", f"{rank} / 11")
            b.metric(f"{SECTORS[selected][0]}の{horizon}日騰落率", f"{row[selected]:+.2f}%")
            c.metric("SPYとの差", f"{data['relative'].loc[asof,selected]:+.2f} pp")
        fig = go.Figure(go.Bar(x=row.values,y=labels,orientation="h",
            marker=dict(color=[SECTORS[s][1] for s in row.index],
                        line=dict(color="black",width=[4 if s == selected else 0 for s in row.index])),
            text=[f"{v:.2f}%" for v in row],textposition="outside",cliponaxis=False))
        style(fig,580)
        low, high = min(0,float(row.min())),max(0,float(row.max()))
        span=max(high-low,1)
        fig.update_xaxes(title=f"{horizon}営業日騰落率（%）",range=[low-span*.2,high+span*.2])
        fig.update_layout(margin=dict(l=230 if selected else 175,r=70,t=35,b=60))
        st.plotly_chart(fig,theme=None,use_container_width=True)
        st.caption(f"順位の計算日：{asof.date()}。指定終了日に欠損がある場合は最後の共通日に戻ります。")
        st.dataframe(pd.DataFrame({"ETF":row.index,"セクター":[SECTORS[s][0] for s in row.index],
            "騰落率_%":row.values,"SPY超過_pp":data["relative"].loc[asof,row.index].values}).sort_values("騰落率_%",ascending=False),
            use_container_width=True,hide_index=True)
    with st.expander("日別の主導セクター・確定データを確認"):
        st.dataframe(leaders,use_container_width=True,hide_index=True)
        st.download_button("主導セクターの推移CSVを保存",leaders.to_csv(index=False).encode("utf-8-sig"),
            file_name="us_sector_leaders.csv",mime="text/csv")
