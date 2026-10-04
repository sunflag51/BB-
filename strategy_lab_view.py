"""Beginner-facing controls, net-profit backtest, and editable earnings hypothesis."""
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from case_study_core import download_prices
from strategy_lab import run_bb_strategy, study_earnings

@st.cache_data(ttl=900,show_spinner=False)
def load_lab_prices(ticker,start,end):
    return download_prices(ticker,pd.Timestamp(start)-pd.Timedelta(days=550),end)


def show_strategy_lab(result,commission_pct,slippage_pct):
    st.divider();st.subheader('6. 仮説を試す研究室')
    st.write('これまでのBB下限ケース分析・市場分析・AI検証に加え、実際の売買ルールの損益と決算仮説を検証します。条件は画面で変えられます。')
    with st.expander('この画面でできること',expanded=True):
        st.markdown('''**① BB下限エントリー:** 下限に近づいた日を確認し、翌営業日の始値で買うルールを検証。Stop・利確・保有期間・手数料を調整し、勝率ではなく手数料後の合計損益と最大資金減少を見る。\n\n**② 決算前仮説:** 確認した四半期決算日を入力し、決算直前までの一定期間で大きく上がったか、ピークから反落したかを見る。\n\n**③ 条件を保存:** 設定と結果を表・CSV・相談用レポートで保存し、仮説を変えて比較する。''')
    idea=st.text_area('今の仮説・次に調べたいこと（自由記入）',key='lab_hypothesis_note',placeholder='例：NVDAは決算前20営業日に一度5%以上上がり、決算直前にピークから3%以上反落する',height=75)
    a,b,c=st.columns(3)
    with a:
        st.number_input('BB下限からの終値許容距離（%）',0.0,20.0,1.0,.25,key='lab_entry_tol',help='前日の終値がBB下限からこの%以内、または日中安値が下限に到達したら候補。翌日の始値で買います。')
    with b:st.number_input('Stop幅（シグナル日のATR倍率）',.1,10.,1.5,.1,key='lab_stop_atr')
    with c:st.number_input('利確幅（Stopリスクの何倍）',.1,10.,2.,.1,key='lab_target_r')
    d,e,f=st.columns(3)
    with d:st.number_input('最大保有営業日',1,252,20,1,key='lab_max_hold')
    with e:st.number_input('1取引で許容する資金リスク（%）',.01,100.,1.,.1,key='lab_risk_pct')
    with f:st.number_input('1銘柄への投資上限（資金比%）',1.,100.,50.,1.,key='lab_alloc_pct')
    g,h=st.columns(2)
    with g:st.number_input('検証開始時資金（株価通貨建て）',100.,1e12,10000.,1000.,key='lab_cash')
    with h:st.caption('米国株は通常USD、日本株はJPY単位の株価通貨で入力。売買コストは上で設定した片道手数料・スリッページを使います。')
    with st.expander('BB期間・幅も変更する（初期値：20日・標準偏差2）'):
        st.number_input('BB計算期間',5,100,20,1,key='lab_bb_period')
        st.number_input('BB標準偏差倍率',.5,5.,2.,.1,key='lab_bb_std')
    settings=(float(st.session_state.lab_entry_tol),float(st.session_state.lab_stop_atr),float(st.session_state.lab_target_r),int(st.session_state.lab_max_hold),float(st.session_state.lab_risk_pct),float(st.session_state.lab_alloc_pct),float(st.session_state.lab_cash),int(st.session_state.lab_bb_period),float(st.session_state.lab_bb_std),float(commission_pct),float(slippage_pct))
    hash_key=hashlib.sha256(repr((result['ticker'],str(result['analysis_start']),str(result['analysis_end']),int(result.get('atr_period',14)),settings)).encode()).hexdigest()[:12]
    if st.button('BB下限ルールの損益を計算',type='primary',key='run_strategy_lab'):
        with st.spinner('株価を取得し、Stop・利確・コスト込みで取引を再計算しています…'):
            prices=load_lab_prices(result['ticker'],result['analysis_start'],result['analysis_end'])
            output=run_bb_strategy(prices,settings[6],settings[0],settings[1],settings[2],settings[3],settings[4],settings[5],settings[9],settings[10],result['analysis_start'],settings[7],settings[8],int(result.get('atr_period',14))) if not prices.empty else {'error':'株価データを取得できませんでした。'}
            st.session_state.lab_backtest={'signature':hash_key,'result':output}
    saved=st.session_state.get('lab_backtest')
    if saved and saved.get('signature')==hash_key:
        back=saved['result']
        if back.get('error'):st.warning(back['error'])
        else:
            s=back['summary'];st.markdown('#### BBルールの結果')
            k1,k2,k3,k4=st.columns(4)
            k1.metric('取引後の合計損益',f"{s['最終損益_通貨']:+,.2f}")
            k2.metric('開始資金に対する損益',f"{s['合計損益_%']:+.2f}%")
            k3.metric('取引数・勝率',f"{s['取引数']}回・{s['勝率_%']:.1f}%" if pd.notna(s['勝率_%']) else f"{s['取引数']}回")
            k4.metric('最大資金減少',f"{s['最大資金減少_%']:.2f}%")
            st.caption(f"株価を期間中ずっと保有した参考騰落率（配当・コスト前）：{s['市場保有参考_%']:+.2f}%。売買コスト片道：手数料{commission_pct:.3f}%＋スリッページ{slippage_pct:.3f}%。決済は損切り優先、期間末は強制決済。損益は株価通貨単位。")
            if s['取引数']==0:st.info('この期間は設定したBB条件で取引がありません。許容距離や期間を見直してください。')
            eq=back['equity']
            fig=go.Figure(go.Scatter(x=eq.index,y=eq['口座評価額_通貨'],name='ルール資産額',mode='lines',line=dict(color='#1565c0',width=2)))
            fig.add_hline(y=settings[6],line_dash='dash',line_color='gray',annotation_text='開始資金')
            fig.update_layout(title='取引コスト後の口座評価額',height=380,plot_bgcolor='white',paper_bgcolor='white',font_color='black',yaxis_title='株価通貨',xaxis_title='日付')
            st.plotly_chart(fig,use_container_width=True,theme=None)
            with st.expander('取引一覧・損益CSV'):
                st.dataframe(back['trades'],hide_index=True,use_container_width=True)
                st.download_button('取引一覧CSV',back['trades'].to_csv(index=False).encode('utf-8-sig'),file_name='bb_strategy_trades.csv',mime='text/csv')
            summary=pd.DataFrame([{'仮説・メモ':idea,'銘柄':result['ticker'],'開始日':result['analysis_start'],'終了日':result['analysis_end'],'ATR期間':int(result.get('atr_period',14)),**dict(zip(['下限許容距離_%','Stop_ATR倍率','利確_R','最大保有日','許容リスク_%','投資上限_%','開始資金','BB期間','BB幅','片道手数料_%','片道スリッページ_%'],settings)),**s}])
            report='BB下限ルール検証レポート\n仮説・メモ: '+idea+'\n条件を変えた場合の取引コスト後の損益を確認してください.\n'+summary.to_csv(index=False,sep='\t',float_format='%.4f')+'\n\n注意：翌営業日始値で買う簡易検証。日中Stopと利確が両方到達した日はStopを先に計上。株数はリスク予算と資金上限で算定。将来利益を保証しません。'
            with st.expander('この仮説・結果をコピーして相談する'):
                st.code(report,language=None)
                st.download_button('仮説と損益をCSV保存',summary.to_csv(index=False).encode('utf-8-sig'),file_name='bb_strategy_summary.csv',mime='text/csv')
    elif saved:st.info('条件が変更されました。古い結果を隠しています。再計算してください。')

    st.markdown('#### NVDA「決算前の上昇後、利益確定売り」の仮説')
    st.write('確認した決算日を入力すると、決算日の前だけを調べます。各決算の直前期間に一度でも上がった幅と、その高値から決算直前終値までの反落を計算します。')
    dates_text=st.text_area('四半期決算日（1行に1日、YYYY-MM-DD）',key='lab_earnings_dates',height=110,placeholder='YYYY-MM-DD\nYYYY-MM-DD\n実際に発表された決算日を入力してください。日付は自動補完しません。')
    q1,q2,q3=st.columns(3)
    with q1:st.number_input('決算前に見る営業日数',2,120,20,1,key='lab_event_days')
    with q2:st.number_input('「大きな上昇」の基準（%）',0.,100.,5.,.5,key='lab_event_rise')
    with q3:st.number_input('ピーク後反落の基準（%）',0.,100.,3.,.5,key='lab_event_pullback')
    event_sig=(str(result['ticker']),str(result['analysis_start']),str(result['analysis_end']),dates_text,int(st.session_state.lab_event_days),float(st.session_state.lab_event_rise),float(st.session_state.lab_event_pullback))
    eh=hashlib.sha256(repr(event_sig).encode()).hexdigest()[:12]
    if st.button('決算前仮説を検証',key='run_earnings_hypothesis'):
        try:
            event_dates=[pd.Timestamp(v.strip()) for v in dates_text.splitlines() if v.strip()]
            with st.spinner('決算前の株価推移を測定しています…'):
                prices=load_lab_prices(result['ticker'],result['analysis_start'],result['analysis_end'])
                output=study_earnings(prices,event_dates,int(st.session_state.lab_event_days),float(st.session_state.lab_event_rise),float(st.session_state.lab_event_pullback)) if not prices.empty else {'error':'株価データを取得できませんでした。'}
                st.session_state.lab_events={'signature':eh,'result':output}
        except Exception as exc:st.error(f'日付を YYYY-MM-DD 形式で確認してください。詳細: {exc}')
    ev=st.session_state.get('lab_events')
    if ev and ev.get('signature')==eh:
        out=ev['result']
        if out.get('error'):st.warning(out['error'])
        elif out['summary']['計算可能決算数']==0:st.info('計算できる決算がありません。決算日を1行ずつ入力してください。')
        else:
            es=out['summary'];v1,v2,v3,v4=st.columns(4)
            v1.metric('計算できた決算数',f"{es['計算可能決算数']} / {es['登録決算数']}")
            v2.metric(f"{st.session_state.lab_event_rise:g}%以上上昇",f"{es['上昇閾値到達数']}回・{es['上昇閾値到達割合_%']:.1f}%")
            v3.metric(f"上昇後、{st.session_state.lab_event_pullback:g}%以上反落",f"{es['利益確定売り仮説に沿う反落数']}回")
            v4.metric('決算前上昇の中央値',f"{es['中央値上昇_%']:+.2f}%")
            st.caption(f"ピーク後反落の中央値：{es['中央値ピーク後反落_%']:+.2f}%。これは値動きの一致度です。利益確定売りが原因だと証明するものではありません。")
            st.dataframe(out['events'],hide_index=True,use_container_width=True)
            e_report='決算前仮説検証レポート\n入力した仮説・メモ: '+idea+'\n検証内容：決算前に大きく上昇し、決算前最終日にピークから反落する。\n'+pd.DataFrame([{'仮説・メモ':idea,'銘柄':result['ticker'],'観察営業日':out['pre_days'],'大幅上昇基準_%':out['rise_threshold_pct'],'反落基準_%':out['pullback_threshold_pct'],**es}]).to_csv(index=False,sep='\t',float_format='%.4f')+'\n\n各行が1回の決算。利益確定が原因だとは証明できず、イベント数が少ない場合は不確実です。'
            with st.expander('決算仮説の結果をコピーして相談する'):
                st.code(e_report,language=None)
                st.download_button('決算ごとの一覧CSV',out['events'].to_csv(index=False).encode('utf-8-sig'),file_name='earnings_hypothesis.csv',mime='text/csv')
    elif ev:st.info('決算日または条件が変わりました。再検証してください。')
