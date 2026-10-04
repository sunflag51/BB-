"""Helpers for saving ticker and analysis-period presets."""
from __future__ import annotations

import pandas as pd

PROFILE_FIELDS = ["保存名", "銘柄コード", "銘柄名", "基準日", "分析開始日", "分析終了日"]


def normalize_analysis_profiles(frame):
    if frame is None:
        return pd.DataFrame(columns=PROFILE_FIELDS)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if frame.empty:
        return pd.DataFrame(columns=PROFILE_FIELDS)
    aliases = {
        "name": "保存名", "profile": "保存名", "ticker": "銘柄コード", "symbol": "銘柄コード",
        "company": "銘柄名", "ticker_name": "銘柄名", "day0": "基準日",
        "start": "分析開始日", "analysis_start": "分析開始日",
        "end": "分析終了日", "analysis_end": "分析終了日",
    }
    x = frame.rename(columns={c: aliases.get(str(c).strip().casefold(), c) for c in frame.columns}).copy()
    for col in PROFILE_FIELDS:
        if col not in x:
            x[col] = ""
    x = x[PROFILE_FIELDS].copy()
    for col in PROFILE_FIELDS:
        x[col] = x[col].fillna("").astype(str).str.strip()
    x["銘柄コード"] = x["銘柄コード"].str.upper()
    for col in ["基準日", "分析開始日", "分析終了日"]:
        parsed = pd.to_datetime(x[col], errors="coerce")
        x[col] = parsed.dt.strftime("%Y-%m-%d").fillna("")
    x = x[(x["保存名"] != "") & (x["銘柄コード"] != "")]
    x = x[(x["基準日"] != "") & (x["分析開始日"] != "") & (x["分析終了日"] != "")]
    x = x.drop_duplicates("保存名", keep="last")
    return x.reset_index(drop=True)


def save_analysis_profile(frame, name, ticker, ticker_name, case_date, start_date, end_date):
    label = str(name or "").strip()
    code = str(ticker or "").strip().upper()
    if not label:
        raise ValueError("保存名を入力してください。")
    if not code:
        raise ValueError("銘柄コードを入力してください。")
    dates = [pd.Timestamp(v).normalize() for v in (case_date, start_date, end_date)]
    if dates[1] > dates[0]:
        raise ValueError("分析開始日は基準日以前にしてください。")
    if dates[2] < dates[0]:
        raise ValueError("分析終了日は基準日以降にしてください。")
    row = dict(zip(PROFILE_FIELDS, [label, code, str(ticker_name or code).strip(), *(d.strftime("%Y-%m-%d") for d in dates)]))
    x = normalize_analysis_profiles(frame)
    x = x[x["保存名"] != label]
    return pd.concat([x, pd.DataFrame([row])], ignore_index=True)[PROFILE_FIELDS]


def delete_analysis_profile(frame, name):
    x = normalize_analysis_profiles(frame)
    return x.loc[x["保存名"] != str(name)].reset_index(drop=True)
