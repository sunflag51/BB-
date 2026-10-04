"""Import/export and validation for named capital-management presets."""
import pandas as pd

PROFILE_FIELDS=['プロファイル名','総資金','1銘柄予算','許容損失率_%','資金通貨','株価通貨','為替取得方法','手入力換算レート']
KEY_FIELDS={'総資金':'total_capital','1銘柄予算':'symbol_budget','許容損失率_%':'risk_pct',
            '資金通貨':'capital_currency','株価通貨':'quote_currency',
            '為替取得方法':'capital_fx_mode','手入力換算レート':'manual_quote_fx'}

def normalize_profiles(data):
    if data is None:return pd.DataFrame(columns=PROFILE_FIELDS)
    if isinstance(data,dict):
        rows=[]
        for name,values in data.items():
            rows.append({'プロファイル名':name,**{label:values.get(key) for label,key in KEY_FIELDS.items()}})
        data=pd.DataFrame(rows)
    x=pd.DataFrame(data).copy()
    if x.empty:return pd.DataFrame(columns=PROFILE_FIELDS)
    if 'プロファイル名' not in x:return pd.DataFrame(columns=PROFILE_FIELDS)
    for col in PROFILE_FIELDS:
        if col not in x:x[col]=None
    x=x[PROFILE_FIELDS]
    x['プロファイル名']=x['プロファイル名'].fillna('').astype(str).str.strip()
    x=x[x['プロファイル名'].ne('')].drop_duplicates('プロファイル名',keep='last')
    for c in ['総資金','1銘柄予算','許容損失率_%','手入力換算レート']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    x['資金通貨']=x['資金通貨'].fillna('JPY').astype(str).str.upper().str.strip()
    x['株価通貨']=x['株価通貨'].fillna('USD').astype(str).str.upper().str.strip()
    x['為替取得方法']=x['為替取得方法'].fillna('自動（ドル円）').astype(str)
    x['手入力換算レート']=x['手入力換算レート'].fillna(1.0)
    return x.reset_index(drop=True)

def profile_values(row):
    return {KEY_FIELDS[label]:row[label] for label in KEY_FIELDS}

def save_profile(data,name,values):
    x=normalize_profiles(data);label=str(name or '').strip()
    if not label:raise ValueError('保存名を入力してください。')
    row={'プロファイル名':label,**{field:values.get(key) for field,key in KEY_FIELDS.items()}}
    if pd.isna(row['総資金']) or row['総資金']<=0:raise ValueError('総資金は0より大きくしてください。')
    if pd.isna(row['1銘柄予算']) or row['1銘柄予算']<=0:raise ValueError('1銘柄予算は0より大きくしてください。')
    if pd.isna(row['許容損失率_%']) or not 0<row['許容損失率_%']<=100:raise ValueError('許容損失率は0より大きく100%以下にしてください。')
    x=x[x['プロファイル名'].ne(label)]
    if x.empty:return normalize_profiles(pd.DataFrame([row]))
    return normalize_profiles(pd.concat([x,pd.DataFrame([row])],ignore_index=True))

def delete_profile(data,name):
    x=normalize_profiles(data)
    return x.loc[x['プロファイル名'].ne(str(name))].reset_index(drop=True)

def conversion_rate(quote_currency, capital_currency, usd_jpy=None, method='自動（ドル円）', manual_rate=None):
    """Return funding-currency units per one quote-currency unit; None means unavailable."""
    quote=str(quote_currency).upper();capital=str(capital_currency).upper()
    if quote==capital:return 1.0
    if method=='手入力':
        try:r=float(manual_rate)
        except (TypeError,ValueError):return None
        return r if r>0 else None
    if {quote,capital}!={'USD','JPY'}:return None
    try:r=float(usd_jpy)
    except (TypeError,ValueError):return None
    if r<=0:return None
    return r if (quote,capital)==('USD','JPY') else 1.0/r
