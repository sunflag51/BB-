"""Rolling chronological comparisons. No holdout-based model selection."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from ai_analysis import build_stock_features, align_past_context, build_targets, CLASSES

VARIANTS = [(False, False, '市場なし・補正なし'), (False, True, '市場なし・補正あり'),
            (True, False, '市場あり・補正なし'), (True, True, '市場あり・補正あり')]


def scores(data):
    actual, pred = data['結果'], data['AI判定']
    up, down = pred.eq('上昇'), actual.eq('下降')
    counts = pred.value_counts().reindex(CLASSES, fill_value=0)
    lead = '・'.join(counts.index[counts.eq(counts.max())])
    recalls = [float(pred[actual.eq(c)].eq(c).mean()) if actual.eq(c).any() else np.nan for c in CLASSES]
    return {'検証日数': len(data), '正解率_%': accuracy_score(actual, pred)*100,
        '多数派基準正解率_%': accuracy_score(actual, data['基準判定'])*100,
        '上昇予測の的中率_%': float(actual[up].eq('上昇').mean())*100 if up.any() else np.nan,
        '下降を捉えた割合_%': float(pred[down].eq('下降').mean())*100 if down.any() else np.nan,
        '3分類平均捕捉率_%': float(np.mean(recalls))*100 if not any(pd.isna(recalls)) else np.nan,
        'Macro_F1': f1_score(actual, pred, labels=CLASSES, average='macro', zero_division=0),
        '最多予測分類': lead, '最多予測割合_%': counts.max()/len(data)*100,
        **{f'{c}予測日数': int(counts[c]) for c in CLASSES}}


def walk_forward(prices, start, end, context=None, horizon=5, threshold=2., atr_period=14,
                 wave_width=3, train_window=126, test_window=42):
    if train_window < 120 or test_window < 1 or horizon < 1 or threshold <= 0:
        return {'error': '学習120日以上・検証1日以上・判定1日以上・判定幅0より大きい値が必要です。'}
    p = prices.sort_index().loc[:pd.Timestamp(end)]
    stock = build_stock_features(p, atr_period, wave_width)
    x = align_past_context(stock, context).loc[pd.Timestamp(start):pd.Timestamp(end)]
    core = list(stock.columns)
    market = [c for c in x if c.startswith('前日')]
    if not market:
        return {'error': '4通りの比較に必要な市場データがありません。取得できる状態で再実行してください。'}
    targets = build_targets(p, horizon, threshold)
    data = x.join(targets).replace([np.inf, -np.inf], np.nan).dropna(subset=core+['結果', '結果確定日'])
    first = train_window + horizon
    if len(data) <= first:
        return {'error': f'比較可能日は{len(data)}日です。学習{train_window}日と未来結果除外分の後に検証日が必要です。期間を広げてください。'}
    records, folds, notes = [], [], []
    for fold, pos in enumerate(range(first, len(data), test_window), 1):
        test = data.iloc[pos:pos+test_window]
        boundary = test.index[0]
        training = data.loc[(data.index < boundary) & (data['結果確定日'] < boundary)].tail(train_window)
        counts = training['結果'].value_counts()
        if len(training) < 120 or len(counts) < 2 or counts.min() < 10:
            notes.append(f'期間{fold}（{boundary.date()}開始）は学習日数・分類例不足のため4通りとも除外。')
            continue
        selected_market = [c for c in market if training[c].notna().mean() >= .95]
        if not selected_market:
            notes.append(f'期間{fold}は学習市場データの充足率95%を満たす特徴がなく、4通りとも除外。')
            continue
        medians = training[selected_market].median()
        majority = next(c for c in CLASSES if counts.get(c, 0) == counts.max())
        audit = {'期間番号': fold, '学習開始': training.index[0], '学習最終特徴日': training.index[-1],
                 '学習最終結果確定日': training['結果確定日'].max(), '検証開始': boundary,
                 '検証終了': test.index[-1], '学習日数': len(training),
                 '市場特徴': ', '.join(selected_market),
                 '市場欠損補完セル数': int(test[selected_market].isna().sum().sum())}
        for use_market, balanced, name in VARIANTS:
            columns = core + (selected_market if use_market else [])
            train_x = training[columns].copy()
            test_x = test[columns].copy()
            if use_market:
                train_x[selected_market] = train_x[selected_market].fillna(medians)
                test_x[selected_market] = test_x[selected_market].fillna(medians)
            model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, int(len(training)*.05)),
                         random_state=42, class_weight='balanced' if balanced else None)
            model.fit(train_x, training['結果'])
            frame = test[['結果', '将来騰落率_%', '結果確定日']].copy()
            frame['AI判定'] = model.predict(test_x)
            frame['基準判定'] = majority
            frame['判断条件番号'] = model.apply(test_x).astype(int)
            frame['比較方法'] = name
            frame['期間番号'] = fold
            records.append(frame)
            folds.append({**audit, '比較方法': name, **scores(frame)})
    if not records:
        return {'error': '条件を満たす検証期間がありません。分析期間を広げてください。', 'notes': notes}
    predictions = pd.concat(records).reset_index(names='日付')
    summary = pd.DataFrame([{'比較方法': name, **scores(predictions.loc[predictions['比較方法'].eq(name)])}
                           for _, _, name in VARIANTS])
    summary['基準との差_pp'] = summary['正解率_%']-summary['多数派基準正解率_%']
    nfold = len({r['期間番号'] for r in folds})
    if nfold < 2: notes.append('検証が1期間だけです。複数時期の安定性は確認できません。')
    return {'error': None, 'summary': summary, 'folds': pd.DataFrame(folds), 'predictions': predictions,
            'notes': notes, 'fold_count': nfold, 'train_window': train_window, 'test_window': test_window}
