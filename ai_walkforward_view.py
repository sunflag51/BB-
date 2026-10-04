"""UI for comparing rolling models without replacing the fixed explanation model."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ai_walkforward import walk_forward, VARIANTS
from market_view import style


def comparison_report(result, comparison, horizon, threshold, width, atr_period, sector):
    table = lambda f: f.to_csv(index=False, sep='\t', float_format='%.4f').rstrip()
    return '\n'.join(['AI比較の相談用レポート v2.7.0',
        '4通りの成績・予測の偏り・時期ごとの安定性を確認してください。',
        f"銘柄: {result['ticker']} / 分析期間: {result['analysis_start']} ～ {result['analysis_end']}",
        f'判定: {horizon}営業日後・±{threshold:g}% / 波: {width} / ATR: {atr_period} / セクター: {sector or "未選択"}',
        f"学習: 直近{comparison['train_window']}日 / 検証: 最大{comparison['test_window']}日ずつ / 期間数: {comparison['fold_count']}",
        '【全期間の比較：同じ検証日】', table(comparison['summary']),
        '【期間ごとの成績・学習境界】', table(comparison['folds']),
        '【直近検証日：各方法20日】', table(comparison['predictions'].groupby('比較方法', sort=False).tail(20)),
        '【注意】', *comparison['notes'],
        '各検証期間の開始前に結果が確定した日だけで学習。検証期間内で学習し直しません。',
        '市場特徴の採否（充足率95%以上）と欠損補完の中央値は各期間の学習データのみで決定。',
        '補正ありは学習分類の頻度に応じた重み付け。成績向上を保証しません。',
        '同じ検証日の4通りの比較。多数派基準は各期間の学習多数派を回答。',
        '割合の単位は%・差はpp。算出不可は対象日0。期間別の判断条件番号は別モデルなので直接比較不可。',
        '後続期間の学習には、結果が確定した以前の検証日も含みます。',
        '日々の5日後などの結果は重なるため独立試行ではありません。比較して選んだ方法には別の未使用期間での検証が必要。',
        'この比較は上部の固定モデルの選択日分類を変更しません。最新日の売買予測を追加する機能ではありません。'])


def show_walkforward(result, bundle, signature, sector, horizon, threshold, width, atr_period, context_loader):
    st.markdown('### 時期をずらしたAI検証：4通りを比較')
    st.write('過去の一定日数で学習し、その後の期間を予測。期間を進めて学習し直し、相場が変わっても分類できるか確認します。')
    a,b = st.columns(2)
    train_window = a.selectbox('比較の学習日数（直近）', [126,252], key='wf_train_days')
    test_window = b.selectbox('一度に検証する日数', [21,42,63], index=1, key='wf_test_days')
    st.caption('市場あり／なし × 分類の偏り補正あり／なしを同じ日付で比較します。上部の市場チェックに関係なく、比較用に市場データを取得します。補正ありは少ない分類を学習で重く扱う方法です。')
    sig = (signature, train_window, test_window)
    if st.button('4通りの時系列比較を実行', key='run_walkforward'):
        with st.spinner('市場データを取得し、期間ごとに4通りを検証しています…'):
            context, notes = context_loader()
            comparison = walk_forward(bundle['prices'], result['analysis_start'], result['analysis_end'], context,
                horizon, threshold, atr_period, width, train_window, test_window)
            comparison['notes'] = comparison.get('notes', []) + notes
            st.session_state['walkforward_bundle'] = {'signature': sig, 'result': comparison}
    saved = st.session_state.get('walkforward_bundle')
    if not saved or saved['signature'] != sig:
        st.info('上の実行ボタンを押してください。条件変更後は再実行が必要です。')
        return
    comparison = saved['result']
    for note in comparison.get('notes', []): st.info(note)
    if comparison.get('error'):
        st.warning(comparison['error'])
        return
    st.write(f"検証期間数：{comparison['fold_count']}。集計は期間別正解率の単純平均ではなく、全検証日の合計です。")
    summary = comparison['summary']
    displayed = summary.copy()
    for col in displayed:
        if col.endswith('_%') or col.endswith('_pp') or col == 'Macro_F1':
            displayed[col] = displayed[col].map(lambda v: f'{v:.2f}' if pd.notna(v) else '算出不可')
    st.dataframe(displayed, hide_index=True, use_container_width=True)
    st.caption('正解率だけでなく、上昇予測の的中率・下降の捕捉・最多予測割合を比較してください。3分類平均捕捉率は各分類の捕捉率の平均で、実際に存在しない分類がある場合は算出不可です。')
    fig = go.Figure()
    for _,_,name in VARIANTS:
        f = comparison['folds'].loc[comparison['folds']['比較方法'].eq(name)]
        fig.add_trace(go.Scatter(x=f['検証開始'], y=f['正解率_%'], mode='lines+markers', name=name))
    style(fig,430)
    fig.update_layout(title='検証開始日ごとの正解率', legend=dict(orientation='h',y=-.28), margin=dict(b=140))
    fig.update_yaxes(title='正解率（%）', range=[0,100]);fig.update_xaxes(title='検証期間の開始日')
    st.plotly_chart(fig, theme=None, use_container_width=True)
    with st.expander('期間ごとの成績と未来データ除外を確認'):
        st.dataframe(comparison['folds'], hide_index=True, use_container_width=True)
    with st.expander('全検証日の4通りの予測・CSV保存'):
        st.dataframe(comparison['predictions'], hide_index=True, use_container_width=True)
        st.download_button('4通りの全検証結果CSV', comparison['predictions'].to_csv(index=False).encode('utf-8-sig'),
            file_name='ai_walkforward_predictions.csv', mime='text/csv')
        st.download_button('期間別の成績CSV', comparison['folds'].to_csv(index=False).encode('utf-8-sig'),
            file_name='ai_walkforward_scores.csv', mime='text/csv')
    st.caption('比較結果で方法を選んだ後は別の未使用期間で確認が必要です。後続の学習には結果が確定した以前の検証日も含みます。隣り合う日の将来期間は重なります。上部の固定モデルによる選択日の分類は、この比較では変更されません。')
    report = comparison_report(result, comparison, horizon, threshold, width, atr_period, sector)
    with st.expander('4通りの比較結果をコピーして相談する', expanded=True):
        st.caption('枠の右上のコピーアイコンを押して、この会話に貼り付けてください。')
        st.code(report, language=None)
        st.download_button('比較相談レポートを保存', report.encode('utf-8-sig'),
            file_name='ai_walkforward_report.txt', mime='text/plain')
