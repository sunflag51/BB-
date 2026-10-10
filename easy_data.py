"""Portable per-session analysis sets. Explicit backup, no shared server storage."""
import json, math
from datetime import date
from io import StringIO
import pandas as pd
from ticker_sheet import normalize_sheet, add_ticker, options_with_saved
from analysis_profiles import normalize_analysis_profiles, parse_analysis_profiles_csv
from capital_profiles import normalize_profiles

DEFAULTS={'cap_total':1000000.,'cap_budget':300000.,'cap_risk_pct':1.,'cap_currency':'JPY',
 'quote_currency':'USD','capital_fx_mode':'自動（ドル円）','manual_quote_fx':1.,
 'easy_atr':14,'easy_multipliers':'1.0,1.5,2.0','easy_selected_atr':1.5,
 'easy_commission':.1,'easy_slippage':.1,'wave_width':3}
DATES=['case_date_value','analysis_start_value','analysis_end_value']
TABLES=['saved_tickers','analysis_profiles','capital_profiles','case_ledger']

def validate_set(r):
 if not isinstance(r,dict):raise ValueError('分析セットの形式が違います。')
 out={'name':str(r.get('name','')).strip(),'ticker':str(r.get('ticker','')).strip().upper(),'company':str(r.get('company',''))}
 if not out['name'] or not out['ticker']:raise ValueError('保存名と銘柄が必要です。')
 for k in DATES:out[k]=date.fromisoformat(str(r[k])).isoformat()
 if not out['analysis_start_value']<=out['case_date_value']<=out['analysis_end_value']:raise ValueError('開始日≦基準日≦終了日を確認してください。')
 limits={'cap_total':(0,1e15),'cap_budget':(0,1e15),'cap_risk_pct':(.01,100),'manual_quote_fx':(.000001,1e9),'easy_atr':(5,100),'easy_selected_atr':(.1,10),'easy_commission':(0,5),'easy_slippage':(0,5),'wave_width':(1,10)}
 for k,v in DEFAULTS.items():
  val=r.get(k,v)
  if k in limits:
   val=float(val);lo,hi=limits[k]
   if not math.isfinite(val) or not lo<=val<=hi:raise ValueError(f'{k}の範囲が正しくありません。')
   if k in ('easy_atr','wave_width'):
    if not val.is_integer():raise ValueError('日数は整数にしてください。')
    val=int(val)
  else:val=str(val)
  out[k]=val
 for k in ('cap_currency','quote_currency'):
  if out[k] not in ('JPY','USD'):raise ValueError('通貨はJPYまたはUSDです。')
 if out['capital_fx_mode'] not in ('自動（ドル円）','手入力'):raise ValueError('換算方法が違います。')
 try:
  mult=[float(v.strip()) for v in out['easy_multipliers'].split(',')]
  if not mult or not all(math.isfinite(v) and v>0 for v in mult):raise ValueError()
 except ValueError:raise ValueError('ATR倍率は正の数をカンマ区切りで入力してください。')
 return out

def capture(state,ticker,company,name):
 r={k:state.get(k,v) for k,v in DEFAULTS.items()}
 r.update({k:state[k].isoformat() for k in DATES})
 return validate_set(dict(r,name=name,ticker=ticker,company=company))

def apply_set(state,r):
 r=validate_set(r)
 state['saved_tickers']=add_ticker(state.get('saved_tickers'),r['ticker'],r['company'] or r['ticker'])[0]
 state['ticker_choice']=next(label for label,code in options_with_saved(state['saved_tickers']).items() if code==r['ticker'])
 for k in DATES:state[k]=date.fromisoformat(r[k])
 for k in DEFAULTS:state[k]=r[k]
 state['quote_currency_ticker']=r['ticker']
 state['current_result']=None;state['current_settings']=None

def backup_bytes(state):
 payload={'format':'bb-analysis-backup','version':1,'sets':state.get('easy_sets',[]),
  'tables':{k:state[k].to_csv(index=False) for k in TABLES if isinstance(state.get(k),pd.DataFrame)}}
 return json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')

def parse_backup(raw):
 x=json.loads(raw.decode('utf-8-sig'))
 if not isinstance(x,dict) or x.get('format')!='bb-analysis-backup' or x.get('version')!=1:raise ValueError('このアプリのバックアップファイルを選んでください。')
 if not isinstance(x.get('sets'),list) or not isinstance(x.get('tables'),dict):raise ValueError('バックアップ内容が不正です。')
 sets=[validate_set(r) for r in x['sets']];tables={}
 for k,text in x['tables'].items():
  if k not in TABLES:continue
  if not isinstance(text,str):raise ValueError('保存表が不正です。')
  tables[k]=pd.read_csv(StringIO(text))
 return sets,tables

def merge_sets(old,new):
 d={r['name']:r for r in old}
 for r in new:
  name=r['name'];i=2
  if name in d and d[name]==r:continue
  while name in d:name=f"{r['name']}（読込{i}）";i+=1
  d[name]=dict(r,name=name)
 return list(d.values())
