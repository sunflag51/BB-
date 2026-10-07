"""Japan index/ETF context and TOPIX-17 rotation; display-only calculations."""
import unicodedata
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

INDICES = {"^N225": "日経平均（指数）", "1306.T": "TOPIX（1306 ETF代用）", "1591.T": "JPX日経400（1591 ETF代用）", "2516.T": "東証グロース250（2516 ETF代用）"}
SECTORS = {'1617.T': ('食品', '#1f77b4'), '1618.T': ('エネルギー資源', '#ff7f0e'), '1619.T': ('建設・資材', '#2ca02c'), '1620.T': ('素材・化学', '#d62728'), '1621.T': ('医薬品', '#9467bd'), '1622.T': ('自動車・輸送機', '#8c564b'), '1623.T': ('鉄鋼・非鉄', '#e377c2'), '1624.T': ('機械', '#bcbd22'), '1625.T': ('電機・精密', '#17becf'), '1626.T': ('情報通信・サービスその他', '#4b0082'), '1627.T': ('電力・ガス', '#008080'), '1628.T': ('運輸・物流', '#c15b00'), '1629.T': ('商社・卸売', '#546e7a'), '1630.T': ('小売', '#b71c1c'), '1631.T': ('銀行', '#00695c'), '1632.T': ('金融（除く銀行）', '#6a1b9a'), '1633.T': ('不動産', '#827717')}

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

def _normalize_sector(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(c for c in value if c.isalnum())


# Yahoo industry labels are mapped narrowly; broad US sector labels are not Japanese industries.
JP_INDUSTRIES = {
    "Semiconductor Equipment & Materials":"1625.T", "Semiconductors":"1625.T",
    "Electronic Components":"1625.T", "Consumer Electronics":"1625.T",
    "Scientific & Technical Instruments":"1625.T", "Computer Hardware":"1625.T",
    "Electronic Gaming & Multimedia":"1626.T", "Software—Application":"1626.T",
    "Software—Infrastructure":"1626.T", "Telecom Services":"1626.T",
    "Auto Manufacturers":"1622.T", "Auto Parts":"1622.T", "Drug Manufacturers—General":"1621.T",
    "Banks—Regional":"1631.T", "Banks—Diversified":"1631.T",
    "Steel":"1623.T", "Chemicals":"1620.T", "Specialty Chemicals":"1620.T",
    "Utilities—Regulated Electric":"1627.T", "Utilities—Regulated Gas":"1627.T",
    "Railroads":"1628.T", "Airlines":"1628.T", "Marine Shipping":"1628.T",
    "Real Estate—Development":"1633.T", "Real Estate Services":"1633.T",
}
JP_KNOWN = {"6857.T":("1625.T","電気機器 → 電機・精密"),
            "7974.T":("1626.T","その他製品 → 情報通信・サービスその他")}

def match_sector(info):
    aliases = {_normalize_sector(k):v for k,v in JP_INDUSTRIES.items()}
    for field in ["industry", "industryDisp", "industryKey"]:
        symbol = aliases.get(_normalize_sector(info.get(field)))
        if symbol:
            return symbol, str(info[field])
    return None, str(info.get("industry") or "日本業種の対応分類なし")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_sector_profile(ticker):
    failures = []
    try:
        info = yf.Ticker(ticker).get_info() or {}
        symbol, raw = match_sector(info)
        if symbol:
            return {"symbol": symbol, "source": "Yahoo Financeの企業情報", "raw": raw}
        failures.append("企業情報に対応するセクターがありません")
    except Exception as exc:
        failures.append("企業情報の取得失敗（" + type(exc).__name__ + "）")
    # Only use an exact ticker match; similar company names must not classify another stock.
    try:
        quotes = yf.Search(ticker, max_results=5, news_count=0, timeout=10).quotes
        for quote in quotes:
            if str(quote.get("symbol", "")).strip().upper() == ticker:
                symbol, raw = match_sector(quote)
                if symbol:
                    return {"symbol": symbol, "source": "Yahoo Financeの銘柄検索（コード一致）", "raw": raw}
        failures.append("コード一致の検索結果にセクター分類がありません")
    except Exception as exc:
        failures.append("銘柄検索の取得失敗（" + type(exc).__name__ + "）")
    # Exceptions are not retained by the successful-profile cache.
    raise RuntimeError(" / ".join(failures))


@st.cache_data(ttl=60, show_spinner=False)
def lookup_stock_sector(ticker):
    ticker = unicodedata.normalize("NFKC", str(ticker)).strip().upper()
    if ticker in JP_KNOWN:
        symbol, raw = JP_KNOWN[ticker]
        return {"symbol":symbol,"source":"主要銘柄の日本業種対応表","raw":raw}
    if ticker in SECTORS:
        return {"symbol": ticker, "source": "選択されたセクターETF", "raw": SECTORS[ticker][0]}
    try:
        return _cached_sector_profile(ticker)
    except Exception as exc:
        return {"symbol": None, "source": "分類を取得できませんでした", "raw": str(exc)}


def _mark_sector_manual(ticker):
    st.session_state[f"jp_sector_manual_{ticker}"] = True


def selected_sector_control(ticker):
    ticker = unicodedata.normalize("NFKC", str(ticker)).strip().upper()
    widget_key = f"jp_chosen_sector_{ticker}"
    manual_key = f"jp_sector_manual_{ticker}"
    force_auto = st.button("セクターを再取得して自動選択", key=f"jp_refresh_sector_{ticker}",
        help="取得し直し、手動指定から自動分類へ戻します。")
    if force_auto:
        lookup_stock_sector.clear()
        _cached_sector_profile.clear()
        st.session_state[manual_key] = False
    with st.spinner("選択銘柄のセクターを確認しています…"):
        classification = lookup_stock_sector(ticker)
    detected = classification.get("symbol") or ""
    options = [""] + list(SECTORS)
    previous = st.session_state.get(widget_key, "")
    if manual_key not in st.session_state:
        # Migrate old selections: preserve a nonempty manual correction, refresh an old empty field.
        st.session_state[manual_key] = bool(previous and previous != detected)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = detected
    elif not st.session_state[manual_key] and detected:
        # A changed selectbox index alone does not update an existing widget's state.
        st.session_state[widget_key] = detected
    elif previous not in options:
        st.session_state[widget_key] = ""
    selected = st.selectbox("選択銘柄の所属セクター（自動判定結果を変更できます）", options,
        format_func=lambda s: "未分類・指定なし" if not s else f"{SECTORS[s][0]} ({s})",
        key=widget_key, on_change=_mark_sector_manual, args=(ticker,))
    st.caption(f"取得コード：{ticker}／分類元：{classification['source']}／取得分類：{classification['raw']}。現在の分類を使用し、過去の分類変更は再現しません。")
    if not detected:
        st.warning("自動分類を取得できませんでした。再取得ボタンを押すか、所属セクターを手動指定してください。")
    if selected:
        if st.session_state[manual_key]:kind = "手動指定"
        elif detected:kind = "自動判定"
        else:kind = "前回の選択を維持（今回の分類は未取得）"
        st.success(f"★ {ticker} → {SECTORS[selected][0]} ({selected})｜{kind}")
        st.caption("★は選択銘柄の所属セクターの位置です。銘柄自体の騰落率や、セクター内での銘柄順位ではありません。日本TOPIX-17業種ETFを比較対象に使用します。")
    else:
        st.info("上の欄でセクターを指定するとグラフに★を表示します。幅広い指数ETFには単一の所属セクターを割り当てません。")
    return selected or None


@st.cache_data(ttl=600, show_spinner=False)
def download_market_prices(start, end):
    symbols = list(dict.fromkeys(list(INDICES) + list(SECTORS) + ["1306.T"]))
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
    if "^N225" not in prices or prices["^N225"].dropna().empty:
        return {"error": "取引日基準となる日経平均を取得できませんでした。再取得してください。"}
    calendar = prices.index[prices["^N225"].notna()]
    p = prices.reindex(calendar).sort_index()
    display = (p.index >= pd.Timestamp(start)) & (p.index <= pd.Timestamp(end))
    if not display.any():
        return {"error": "指定期間に日本市場の取引日データがありません。"}
    market = p.reindex(columns=list(INDICES))
    returns = trailing_return(market, horizon).loc[display]
    common = market.loc[display].dropna()
    normalized = common.div(common.iloc[0]).mul(100) if not common.empty else common
    # No partial-universe average when any of the four indices is unavailable.
    score = returns.mean(axis=1).where(returns.notna().all(axis=1))
    sector_prices = p.reindex(columns=list(SECTORS))
    sectors = trailing_return(sector_prices, horizon).loc[display]
    spy = trailing_return(p.reindex(columns=["1306.T"]), horizon)["1306.T"].loc[display]
    relative = sectors.sub(spy, axis=0)
    leaders = []
    for dt, row in sectors.iterrows():
        if not row.notna().all():
            leaders.append({"日付": dt, "ETF": "", "主導セクター": "データ不足", "騰落率_%": np.nan, "TOPIX超過_pp": np.nan, "状態": "17業種すべてが揃わないため未判定"})
            continue
        maximum = float(row.max())
        winners = row.index[np.isclose(row.values, maximum, rtol=0, atol=1e-10)]
        if len(winners) != 1:
            leaders.append({"日付": dt, "ETF": "", "主導セクター": "同率首位", "騰落率_%": maximum, "TOPIX超過_pp": np.nan, "状態": "同率首位: " + " / ".join(winners)})
            continue
        symbol = winners[0]
        leaders.append({"日付": dt, "ETF": symbol, "主導セクター": SECTORS[symbol][0], "騰落率_%": maximum,
            "TOPIX超過_pp": relative.loc[dt, symbol], "状態": "相対首位・全セクター下落" if maximum < 0 else "騰落率首位"})
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
    st.subheader("市場の動向・セクターローテーション（日本市場）")
    st.caption("日本の4指標（日経平均指数＋3つの指数連動ETF代用）とTOPIX-17業種ETFで計算します。売買・Stopの計算には使用しません。")
    ticker = str(result["ticker"]).upper()
    st.caption("日本株コード（.T）から自動切替。日本業種の自動対応は企業情報の業種名を使用し、不明な銘柄は手動指定できます。市場・業種の参考情報です。過去の波の条件分析とは別に表示します。")
    selected = selected_sector_control(ticker)
    horizon = st.selectbox("市場・セクターの比較期間（営業日）", [5, 20, 60], index=1, key="jp_market_horizon")
    if st.button("市場・セクターデータを再取得", key="jp_refresh_market"):
        download_market_prices.clear()
    start, end = result["analysis_start"], result["analysis_end"]
    with st.spinner("日本の市場指標と17業種の価格を取得しています…"):
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
        st.write("指数比較＝同じ基準日の終値を100として、その日の終値÷基準日終値×100。基準日は表示期間内で4指標が揃う最初の日。")
        st.write(f"{horizon}営業日騰落率＝（当日終値÷{horizon}営業日前終値−1）×100。市場スコア＝4指標の騰落率の単純平均。")
        st.write("主導セクター＝17業種ETFの騰落率の首位。同率首位・欠損は灰色。全セクター下落時の首位は『相対首位』です。")
        st.write("セクターは分配金・分割調整後の価格を使用。TOPIX超過＝セクター騰落率−TOPIX ETF騰落率（単位pp）。日経平均は価格指数、その他は分配金調整後ETFの代用で、収益定義が異なります。市場スコアは参考値です。")
        st.write("取引日は日経平均の観測日を使用。各日の判定はその日以前のデータだけで計算。欠損のある期間は未判定。")

    st.markdown("#### 市場グラフ：4指標を同じ日から100で比較")
    norm = data["normalized"]
    if norm.empty:
        st.info("4指標が同じ日に揃うデータがなく、100基準比較は表示できません。")
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
    st.markdown(f"#### 市場スコア：4指標の{horizon}営業日騰落率平均")
    if not valid_score.empty:
        dates = valid_score.index.strftime("%Y-%m-%d").tolist()
        fig = go.Figure(go.Scatter(x=dates, y=valid_score, mode="lines", name="4指標平均騰落率"))
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
    style(fig,820)
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
    band_rows = ["全17業種の首位"]
    band_values = [[codes.get(s,0) for s in leaders["ETF"]]]
    leader_details = leaders[["主導セクター", "ETF", "騰落率_%", "TOPIX超過_pp", "状態"]].to_numpy()
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
        hovertemplate="%{x}<br>%{y}<br>全体首位: %{customdata[0]} (%{customdata[1]})<br>首位騰落率: %{customdata[2]:.2f}%<br>%{customdata[4]}<br>選択セクター騰落率: %{customdata[5]:.2f}%<br>選択セクター順位: %{customdata[6]:.0f} / 17<extra></extra>"))
    # Separate legend entries keep sector colors identifiable without overcrowding the strip.
    for symbol,(name,color) in SECTORS.items():
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(color=color,size=12),name=f"★ {ticker} → {name} ({symbol})" if symbol == selected else f"{name} ({symbol})"))
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(color=palette[0],size=12),name="未判定・同率首位"))
    style(fig,500 if selected else 460)
    fig.update_layout(hovermode="closest",margin=dict(l=90,r=25,t=25,b=240),
        legend=dict(orientation="h",y=-.5,yanchor="top",x=0,font=dict(size=12)))
    fig.update_xaxes(type="category",categoryorder="array",categoryarray=dates,nticks=10)
    fig.update_yaxes(showgrid=False,autorange="reversed",automargin=True)
    day0_line(fig,dates,result["day0"])
    st.plotly_chart(fig,theme=None,use_container_width=True)
    if selected:
        st.caption(f"下の★行は{ticker}の所属セクターが単独首位の日だけ色を付けます。灰色は首位以外・データ不足・同率首位です。日付にカーソルを置くと、その日の17業種中の順位も確認できます。")
    st.caption("色はセクターの種類です。日付にカーソルを置くとセクター名・騰落率・TOPIX ETFとの差を表示します。灰色は17業種不足または同率首位です。")

    complete = sectors.dropna()
    st.markdown("#### 最新共通日のセクター順位")
    if complete.empty:
        st.info("17業種すべてを比較できる日がありません。主導セクターは未判定です。")
    else:
        asof = complete.index[-1]
        row = complete.loc[asof].sort_values()
        labels = [f"★ {ticker} → {SECTORS[s][0]} ({s})" if s == selected else f"{SECTORS[s][0]} ({s})" for s in row.index]
        if selected:
            rank = int(complete.loc[asof].rank(ascending=False,method="min")[selected])
            a,b,c = st.columns(3)
            a.metric(f"★ {ticker}の所属セクター順位", f"{rank} / 17")
            b.metric(f"{SECTORS[selected][0]}の{horizon}日騰落率", f"{row[selected]:+.2f}%")
            c.metric("TOPIX ETFとの差", f"{data['relative'].loc[asof,selected]:+.2f} pp")
        fig = go.Figure(go.Bar(x=row.values,y=labels,orientation="h",
            marker=dict(color=[SECTORS[s][1] for s in row.index],
                        line=dict(color="black",width=[4 if s == selected else 0 for s in row.index])),
            text=[f"{v:.2f}%" for v in row],textposition="outside",cliponaxis=False))
        style(fig,780)
        low, high = min(0,float(row.min())),max(0,float(row.max()))
        span=max(high-low,1)
        fig.update_xaxes(title=f"{horizon}営業日騰落率（%）",range=[low-span*.2,high+span*.2])
        fig.update_layout(margin=dict(l=230 if selected else 175,r=70,t=35,b=60))
        st.plotly_chart(fig,theme=None,use_container_width=True)
        st.caption(f"順位の計算日：{asof.date()}。指定終了日に欠損がある場合は最後の共通日に戻ります。")
        st.dataframe(pd.DataFrame({"ETF":row.index,"セクター":[SECTORS[s][0] for s in row.index],
            "騰落率_%":row.values,"TOPIX超過_pp":data["relative"].loc[asof,row.index].values}).sort_values("騰落率_%",ascending=False),
            use_container_width=True,hide_index=True)
    with st.expander("日別の主導セクター・確定データを確認"):
        st.dataframe(leaders,use_container_width=True,hide_index=True)
        st.download_button("主導セクターの推移CSVを保存",leaders.to_csv(index=False).encode("utf-8-sig"),
            file_name="jp_sector_leaders.csv",mime="text/csv")
