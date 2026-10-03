from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

APP_VERSION = "1.0.0"
BB_PERIOD = 20
BB_STD = 2.0
BW_LOOKBACK = 125
CASE_WINDOW_DAYS = 3
DEFAULT_COMMISSION = 0.001
DEFAULT_SLIPPAGE = 0.001
HORIZONS = (5, 10, 20)


def _clean_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper()


def download_prices(ticker: str, start_date, end_date) -> pd.DataFrame:
    ticker = _clean_ticker(ticker)
    if not ticker:
        return pd.DataFrame()
    s = pd.Timestamp(start_date).normalize()
    e = pd.Timestamp(end_date).normalize()
    if e < s:
        return pd.DataFrame()
    try:
        df = yf.download(
            ticker,
            start=s.strftime("%Y-%m-%d"),
            end=(e + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            return pd.DataFrame()
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["BB_Middle"] = df["Close"].rolling(BB_PERIOD).mean()
    df["BB_Std"] = df["Close"].rolling(BB_PERIOD).std(ddof=0)
    df["BB_Upper"] = df["BB_Middle"] + BB_STD * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - BB_STD * df["BB_Std"]
    df["BandWidth"] = np.where(
        df["BB_Middle"].ne(0),
        (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"],
        np.nan,
    )
    bw_min = df["BandWidth"].rolling(BW_LOOKBACK).min()
    bw_max = df["BandWidth"].rolling(BW_LOOKBACK).max()
    denom = bw_max - bw_min
    df["Normalized_BandWidth"] = np.where(
        denom.ne(0), (df["BandWidth"] - bw_min) / denom, np.nan
    )
    df["Low_BandWidth_Zone"] = df["Normalized_BandWidth"].le(0.20)
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA50_Deviation_Pct"] = (df["Close"] / df["MA50"] - 1.0) * 100.0
    df["Return_20D_Pct"] = df["Close"].pct_change(20) * 100.0
    daily_ret = df["Close"].pct_change()
    df["Vol_20D_Annualized_Pct"] = daily_ret.rolling(20).std(ddof=0) * np.sqrt(252) * 100.0
    df["Close_to_Lower_Pct"] = (df["Close"] / df["BB_Lower"] - 1.0) * 100.0
    df["Low_to_Lower_Pct"] = (df["Low"] / df["BB_Lower"] - 1.0) * 100.0
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Prev_High"] = df["High"].shift(1)
    df["Higher_Low"] = df["Low"] > df["Prev_Low"]
    df["Close_Up"] = df["Close"] > df["Prev_Close"]
    df["Decline_Stop"] = df["Higher_Low"] & df["Close_Up"]
    df["Rebound_Start"] = df["Close"] > df["Prev_High"]
    return df


def resolve_case_date(df: pd.DataFrame, requested_date) -> tuple[pd.Timestamp | None, str]:
    if df.empty:
        return None, "価格データなし"
    d = pd.Timestamp(requested_date).normalize()
    eligible = df.index[df.index >= d]
    if len(eligible) == 0:
        return None, "入力日以降の営業日データなし"
    actual = eligible[0]
    if actual == d:
        return actual, "入力日を使用"
    return actual, f"入力日は非取引日のため次営業日 {actual.date()} を使用"


def case_day0_summary(df: pd.DataFrame, day0: pd.Timestamp) -> pd.DataFrame:
    r = df.loc[day0]
    low_touch = bool(pd.notna(r["BB_Lower"]) and r["Low"] <= r["BB_Lower"])
    close_below = bool(pd.notna(r["BB_Lower"]) and r["Close"] <= r["BB_Lower"])
    if close_below:
        bb_state = "終値がBB下限以下"
    elif low_touch:
        bb_state = "日中安値がBB下限到達・終値は上"
    else:
        bb_state = "BB下限未到達（距離を確認）"
    return pd.DataFrame([{
        "ケース日": day0.date(),
        "Open": r["Open"], "High": r["High"], "Low": r["Low"], "Close": r["Close"],
        "BB下限": r["BB_Lower"], "BB中央": r["BB_Middle"], "BB上限": r["BB_Upper"],
        "終値-BB下限距離_%": r["Close_to_Lower_Pct"],
        "安値-BB下限距離_%": r["Low_to_Lower_Pct"],
        "BB状態": bb_state,
        "BandWidth": r["BandWidth"],
        "正規化BandWidth": r["Normalized_BandWidth"],
        "低BandWidth帯": bool(r["Low_Bandwidth_Zone"]) if "Low_Bandwidth_Zone" in r.index else bool(r["Low_BandWidth_Zone"]),
        "過去20日年率Vol_%": r["Vol_20D_Annualized_Pct"],
        "20日騰落率_%": r["Return_20D_Pct"],
        "MA50乖離_%": r["MA50_Deviation_Pct"],
    }])


def find_signals(df: pd.DataFrame, day0: pd.Timestamp) -> pd.DataFrame:
    try:
        p0 = int(df.index.get_loc(day0))
    except Exception:
        return pd.DataFrame()
    end = min(len(df) - 1, p0 + CASE_WINDOW_DAYS)
    rows = []
    for p in range(p0, end + 1):
        r = df.iloc[p]
        rows.append({
            "Day": p - p0,
            "日付": df.index[p].date(),
            "安値": r["Low"],
            "終値": r["Close"],
            "前日安値": r["Prev_Low"],
            "前日終値": r["Prev_Close"],
            "前日高値": r["Prev_High"],
            "Higher_Low": bool(r["Higher_Low"]),
            "Close_Up": bool(r["Close_Up"]),
            "下落停止": bool(r["Decline_Stop"]),
            "反発開始": bool(r["Rebound_Start"]),
        })
    return pd.DataFrame(rows)


def _first_signal_position(df: pd.DataFrame, day0: pd.Timestamp, signal_col: str):
    p0 = int(df.index.get_loc(day0))
    end = min(len(df) - 1, p0 + CASE_WINDOW_DAYS)
    for p in range(p0, end + 1):
        if bool(df.iloc[p][signal_col]):
            return p
    return None


def _first_hit(df, entry_pos, stop, target, horizon):
    last = entry_pos + horizon - 1
    if last >= len(df):
        return {"Outcome": "将来データ不足", "Outcome_Date": pd.NaT, "Exit_Price": np.nan,
                "Exit_Type": "データ不足", "Gross_R": np.nan, "MFE_R": np.nan, "MAE_R": np.nan,
                "Timeout_Close_R": np.nan}
    entry = float(df.iloc[entry_pos]["Open"])
    risk = entry - stop
    highs, lows = [], []
    for p in range(entry_pos, last + 1):
        r = df.iloc[p]
        o, h, l = float(r["Open"]), float(r["High"]), float(r["Low"])
        highs.append(h); lows.append(l)
        dt = df.index[p]
        if o <= stop:
            exit_price = o
            return {"Outcome": "Stop先着", "Outcome_Date": dt, "Exit_Price": exit_price,
                    "Exit_Type": "StopギャップOpen決済" if o < stop else "Stop決済",
                    "Gross_R": (exit_price-entry)/risk,
                    "MFE_R": (max(highs)-entry)/risk, "MAE_R": (entry-min(lows))/risk,
                    "Timeout_Close_R": np.nan}
        if o >= target:
            exit_price = o
            return {"Outcome": "Target先着", "Outcome_Date": dt, "Exit_Price": exit_price,
                    "Exit_Type": "TargetギャップOpen決済" if o > target else "Target決済",
                    "Gross_R": (exit_price-entry)/risk,
                    "MFE_R": (max(highs)-entry)/risk, "MAE_R": (entry-min(lows))/risk,
                    "Timeout_Close_R": np.nan}
        hit_stop, hit_target = l <= stop, h >= target
        if hit_stop and hit_target:
            return {"Outcome": "同日両方到達・順序不明", "Outcome_Date": dt, "Exit_Price": np.nan,
                    "Exit_Type": "順序不明", "Gross_R": np.nan,
                    "MFE_R": (max(highs)-entry)/risk, "MAE_R": (entry-min(lows))/risk,
                    "Timeout_Close_R": np.nan}
        if hit_stop:
            return {"Outcome": "Stop先着", "Outcome_Date": dt, "Exit_Price": stop,
                    "Exit_Type": "Stop決済", "Gross_R": -1.0,
                    "MFE_R": (max(highs)-entry)/risk, "MAE_R": (entry-min(lows))/risk,
                    "Timeout_Close_R": np.nan}
        if hit_target:
            return {"Outcome": "Target先着", "Outcome_Date": dt, "Exit_Price": target,
                    "Exit_Type": "Target決済", "Gross_R": 2.0,
                    "MFE_R": (max(highs)-entry)/risk, "MAE_R": (entry-min(lows))/risk,
                    "Timeout_Close_R": np.nan}
    close = float(df.iloc[last]["Close"])
    return {"Outcome": "期間内未到達", "Outcome_Date": df.index[last], "Exit_Price": close,
            "Exit_Type": "期間末終値決済", "Gross_R": (close-entry)/risk,
            "MFE_R": (max(highs)-entry)/risk, "MAE_R": (entry-min(lows))/risk,
            "Timeout_Close_R": (close-entry)/risk}


def _net_r(entry, exit_price, risk, commission, slippage):
    buy = entry * (1.0 + slippage)
    sell = exit_price * (1.0 - slippage)
    pnl = sell - buy - buy * commission - sell * commission
    return pnl / risk


def analyze_signal(df: pd.DataFrame, day0: pd.Timestamp, signal_col: str, label: str,
                   commission: float, slippage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _first_signal_position(df, day0, signal_col)
    if p is None:
        return pd.DataFrame([{"シグナル": label, "状態": "Day0～Day3に成立なし"}]), pd.DataFrame()
    if p + 1 >= len(df):
        return pd.DataFrame([{"シグナル": label, "状態": "成立したが翌営業日データなし"}]), pd.DataFrame()
    p0 = int(df.index.get_loc(day0))
    signal_date = df.index[p]
    entry_pos = p + 1
    entry_date = df.index[entry_pos]
    entry = float(df.iloc[entry_pos]["Open"])
    stop = float(df.iloc[p0:p+1]["Low"].min())
    risk = entry - stop
    if risk <= 0:
        return pd.DataFrame([{
            "シグナル": label, "状態": "R計算不可（Entry≦Stop）",
            "シグナル日": signal_date.date(), "Entry日": entry_date.date(),
            "Entry": entry, "Stop": stop
        }]), pd.DataFrame()
    t15, t20 = entry + 1.5*risk, entry + 2.0*risk
    design = pd.DataFrame([{
        "シグナル": label, "状態": "R計算可能",
        "シグナル日": signal_date.date(), "Entry日": entry_date.date(),
        "Entry_Open": entry, "Stop": stop, "1R": risk,
        "1R_%": risk/entry*100.0, "+1.5R": t15, "+2R": t20
    }])
    outcomes = []
    for h in HORIZONS:
        hit = _first_hit(df, entry_pos, stop, t20, h)
        net = np.nan
        if pd.notna(hit["Exit_Price"]) and pd.notna(hit["Gross_R"]):
            net = _net_r(entry, float(hit["Exit_Price"]), risk, commission, slippage)
        outcomes.append({
            "シグナル": label, "期間_営業日": h, "結果": hit["Outcome"],
            "結果日": hit["Outcome_Date"].date() if pd.notna(hit["Outcome_Date"]) else None,
            "決済方法": hit["Exit_Type"], "Gross_R": hit["Gross_R"],
            "Net_R": net, "MFE_R": hit["MFE_R"], "MAE_R": hit["MAE_R"],
            "期間末Close_R": hit["Timeout_Close_R"],
        })
    return design, pd.DataFrame(outcomes)


def build_path_table(df: pd.DataFrame, day0: pd.Timestamp, forward_days: int = 25) -> pd.DataFrame:
    p0 = int(df.index.get_loc(day0))
    end = min(len(df), p0 + forward_days + 1)
    part = df.iloc[p0:end].copy()
    if part.empty:
        return pd.DataFrame()
    base = float(part.iloc[0]["Close"])
    out = pd.DataFrame({
        "営業日": range(len(part)),
        "日付": part.index.date,
        "Open": part["Open"].values,
        "High": part["High"].values,
        "Low": part["Low"].values,
        "Close": part["Close"].values,
        "BB_Lower": part["BB_Lower"].values,
        "BB_Middle": part["BB_Middle"].values,
        "BB_Upper": part["BB_Upper"].values,
        "基準日終値比_%": (part["Close"].values/base - 1.0)*100.0,
    })
    return out


def run_case_study(ticker: str, requested_date, commission=DEFAULT_COMMISSION,
                   slippage=DEFAULT_SLIPPAGE, history_days=550, future_days=120):
    req = pd.Timestamp(requested_date).normalize()
    start = req - pd.Timedelta(days=history_days)
    end = min(pd.Timestamp.today().normalize(), req + pd.Timedelta(days=future_days))
    raw = download_prices(ticker, start, end)
    if raw.empty:
        return {"error": "株価データを取得できませんでした。銘柄コードと日付を確認してください。"}
    df = add_indicators(raw)
    day0, note = resolve_case_date(df, req)
    if day0 is None:
        return {"error": note}
    day0_summary = case_day0_summary(df, day0)
    signal_window = find_signals(df, day0)
    ds_design, ds_outcomes = analyze_signal(
        df, day0, "Decline_Stop", "下落停止", commission, slippage
    )
    rb_design, rb_outcomes = analyze_signal(
        df, day0, "Rebound_Start", "反発開始", commission, slippage
    )
    path = build_path_table(df, day0, 25)
    return {
        "error": None, "ticker": _clean_ticker(ticker), "requested_date": req,
        "day0": day0, "date_note": note, "prices": df,
        "day0_summary": day0_summary, "signal_window": signal_window,
        "design": pd.concat([ds_design, rb_design], ignore_index=True),
        "outcomes": pd.concat([ds_outcomes, rb_outcomes], ignore_index=True),
        "path": path,
    }
