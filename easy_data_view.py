import pandas as pd
import streamlit as st
from io import StringIO
from easy_data import *
from capital_profiles import PROFILE_FIELDS as CAPITAL_FIELDS
from analysis_profiles import PROFILE_FIELDS as ANALYSIS_FIELDS
from case_study_core import normalize_case_ledger, merge_case_ledgers, LEDGER_COLUMNS


def notice(text):st.session_state['easy_notice']=text

def load_selected():
 try:
  r=next(r for r in st.session_state.easy_sets if r['name']==st.session_state.easy_choice)
  apply_set(st.session_state,r);notice(f"「{r['name']}」を呼び出しました。銘柄・期間・資金・分析設定を反映しました。下の分析ボタンを押してください。")
 except Exception as e:notice(f'呼び出せませんでした：{e}')

def restore_file():
 try:
  file=st.session_state.easy_upload
  if file is None:raise ValueError('ファイルを選んでください。')
  raw=file.getvalue();tables={};sets=[]
  if file.name.lower().endswith('.json'):
   sets,tables=parse_backup(raw)
  else:
   frame=pd.read_csv(StringIO(raw.decode('utf-8-sig')))
   if set(ANALYSIS_FIELDS)<=set(frame.columns):
    restored,skipped=parse_analysis_profiles_csv(StringIO(raw.decode('utf-8-sig')))
    if skipped:raise ValueError(f'無効な分析条件が{skipped}件あります。元のCSVを確認してください。')
    tables['analysis_profiles']=restored
    for _,r in restored.iterrows():
     settings={k:st.session_state.get(k,v) for k,v in DEFAULTS.items()}
     settings.update({k:r[c] for k,c in zip(DATES,['基準日','分析開始日','分析終了日'])})
     sets.append(validate_set(dict(settings,name=r['保存名'],ticker=r['銘柄コード'],company=r['銘柄名'])))
   elif set(CAPITAL_FIELDS)<=set(frame.columns):tables['capital_profiles']=frame
   elif set(LEDGER_COLUMNS)<=set(frame.columns):tables['case_ledger']=frame
   elif {'銘柄コード','銘柄名'}<=set(frame.columns):tables['saved_tickers']=frame
   else:raise ValueError('対応する保存CSVではありません。')
  # Validate all tables before changing state; merge preserves existing rows.
  funcs={'saved_tickers':normalize_sheet,'analysis_profiles':normalize_analysis_profiles,'capital_profiles':normalize_profiles,'case_ledger':normalize_case_ledger}
  required={'saved_tickers':['銘柄コード','銘柄名'],'analysis_profiles':ANALYSIS_FIELDS,'capital_profiles':CAPITAL_FIELDS,'case_ledger':LEDGER_COLUMNS}
  updates={}
  for k,frame in tables.items():
   if not set(required[k])<=set(frame.columns):raise ValueError(f'{k}の列が不足しています。')
   if k=='case_ledger':updates[k]=merge_case_ledgers(st.session_state[k],frame)
   else:
    # Keep existing named rows on conflicts; imported names can be added separately.
    combined=pd.concat([frame,st.session_state[k]],ignore_index=True)
    updates[k]=funcs[k](combined)
  newsets=merge_sets(st.session_state.easy_sets,sets)
  for r in newsets:
   updates['saved_tickers']=add_ticker(updates.get('saved_tickers',st.session_state.saved_tickers),r['ticker'],r['company'])[0]
  st.session_state.update(updates);st.session_state.easy_sets=newsets
  notice(f"追加読み込みしました。分析セットは合計{len(newsets)}件です。以前の資金設定・台帳は詳細管理から確認できます。")
 except Exception as e:notice(f'読み込めませんでした。保存データは変更していません：{e}')

def delete_selected():
 if st.session_state.get('easy_delete_confirm'):
  name=st.session_state.easy_choice
  st.session_state.easy_sets=[r for r in st.session_state.easy_sets if r['name']!=name]
  notice(f'「{name}」を一覧から削除しました。バックアップファイルは変更しません。')
 else:notice('削除する場合は確認欄にチェックを入れてください。')

def show_data_home():
 st.session_state.setdefault('easy_sets',[])
 st.session_state.setdefault('saved_tickers',pd.DataFrame(columns=['銘柄コード','銘柄名']))
 st.subheader('保存・呼び出しはここだけ')
 st.caption('①下で銘柄・期間・資金を入力 → ②分析セットに保存 → ③次回は名前を選んで呼び出す')
 if st.session_state.get('easy_notice'):st.info(st.session_state.pop('easy_notice'))
 names=[r['name'] for r in st.session_state.easy_sets]
 if names:
  if st.session_state.get('easy_choice') not in names:st.session_state.easy_choice=names[0]
  st.selectbox('保存した分析セット',names,key='easy_choice')
  st.button('このセットを呼び出す',on_click=load_selected,type='primary')
  with st.expander('保存セットの内容・削除'):
   st.dataframe(pd.DataFrame(st.session_state.easy_sets)[['name','ticker','company','analysis_start_value','analysis_end_value','cap_total','cap_currency']],hide_index=True,use_container_width=True)
   st.checkbox('選んだセットを削除する',key='easy_delete_confirm')
   st.button('選んだセットを削除',on_click=delete_selected)
 else:st.info('まだ分析セットがありません。入力欄の下にある「分析セットに保存」から追加してください。')
 st.download_button('全部まとめてバックアップを保存',backup_bytes(st.session_state),file_name='stock_analysis_backup.json',mime='application/json',key='easy_backup')
 st.caption('再起動・別端末で使うときは、この1ファイルを下から読み込むだけです。設定を追加・変更した後は最新のバックアップを保存してください。株価は分析時に再取得します。')
 with st.expander('バックアップ・以前のCSVを読み込む'):
  st.file_uploader('ファイルを選択（例：stock_analysis_backup.json / analysis_profiles.csv / saved_tickers.csv / capital_profiles.csv / bb_case_ledger_v2_4.csv）',type=['json','csv'],key='easy_upload')
  st.button('このファイルを追加読み込み',on_click=restore_file)
  st.caption('既存の一覧は消しません。異なる内容の同名分析セットは別名で追加します。以前の期間CSVには現在の資金設定を組み合わせます。')
 return st.checkbox('以前の個別CSV管理・比較台帳を開く（通常は不要）',key='easy_legacy')

def show_save_set(ticker,company):
 st.subheader('現在の入力をまとめて保存')
 name=st.text_input('分析セットの名前',placeholder=f'{ticker} 直近半年',key='easy_name')
 overwrite=st.checkbox('同じ名前の分析セットを更新する',key='easy_overwrite')
 if st.button('分析セットに保存',type='primary'):
  try:
   r=capture(st.session_state,ticker,company,name)
   old=st.session_state.easy_sets
   if any(x['name']==r['name'] for x in old) and not overwrite:raise ValueError('同名のセットがあります。名前を変えるか、更新にチェックしてください。')
   st.session_state.easy_sets=[x for x in old if x['name']!=r['name']]+[r]
   st.session_state.saved_tickers=add_ticker(st.session_state.saved_tickers,ticker,company)[0]
   notice('分析セットを保存しました。上の「全部まとめてバックアップを保存」で端末にも残してください。')
   st.rerun()
  except ValueError as e:st.error(str(e))
