"""Watchlist helpers; Streamlit UI state is stored per browser session."""
import pandas as pd

PRESET_TICKERS = {
    'COST': 'コストコ',
    'NVDA': 'エヌビディア',
    'GOOG': 'Alphabet / Google',
    'MSFT': 'Microsoft',
    'AAPL': 'Apple',
    'ISRG': 'Intuitive Surgical',
    '7974.T': '任天堂',
    '6857.T': 'アドバンテスト',
}
COLUMNS = ['銘柄コード', '銘柄名']

def normalize_ticker(value):
    return str(value or '').strip().upper()

def normalize_sheet(frame):
    if frame is None or frame.empty:return pd.DataFrame(columns=COLUMNS)
    x=frame.copy()
    aliases={'ticker':'銘柄コード','symbol':'銘柄コード','code':'銘柄コード','name':'銘柄名'}
    x=x.rename(columns={c:aliases.get(str(c).strip().casefold(),c) for c in x.columns})
    if '銘柄コード' not in x:return pd.DataFrame(columns=COLUMNS)
    if '銘柄名' not in x:x['銘柄名']=''
    x=x[COLUMNS].copy()
    x['銘柄コード']=x['銘柄コード'].map(normalize_ticker)
    x['銘柄名']=x['銘柄名'].fillna('').astype(str).str.strip()
    x=x[x['銘柄コード'].ne('')].drop_duplicates('銘柄コード',keep='last')
    x.loc[x['銘柄名'].eq(''),'銘柄名']=x.loc[x['銘柄名'].eq(''),'銘柄コード']
    return x.reset_index(drop=True)

def add_ticker(frame, ticker, name=None):
    x=normalize_sheet(frame);code=normalize_ticker(ticker)
    if not code:return x,False
    label=str(name or PRESET_TICKERS.get(code,code)).strip() or code
    if x['銘柄コード'].eq(code).any():
        x.loc[x['銘柄コード'].eq(code),'銘柄名']=label
        return x,False
    return pd.concat([x,pd.DataFrame([{'銘柄コード':code,'銘柄名':label}])],ignore_index=True),True

def remove_tickers(frame, tickers):
    x=normalize_sheet(frame);codes={normalize_ticker(v) for v in tickers}
    return x.loc[~x['銘柄コード'].isin(codes)].reset_index(drop=True)

def options_with_saved(frame):
    x=normalize_sheet(frame)
    options={f'{code}（{name}）':code for code,name in PRESET_TICKERS.items()}
    for _,row in x.iterrows():
        label=f"{row['銘柄コード']}（{row['銘柄名']}）"
        options[label]=row['銘柄コード']
    options['その他（銘柄コードを入力）']='__CUSTOM__'
    return options
