"""Interpretable stock direction analysis. No order execution or trading-rule changes."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

CLASSES = ["上昇", "横ばい", "下降"]
DEFINITIONS = {
    "50日線乖離_%": "終値÷50日平均−1（%）",
    "200日線乖離_%": "終値÷200日平均−1（%）",
    "50日線の5日変化_%": "50日平均÷5営業日前の50日平均−1（%）",
    "200日線の20日変化_%": "200日平均÷20営業日前の200日平均−1（%）",
    "過去1日騰落率_%": "終値の1営業日変化（%）",
    "過去5日騰落率_%": "終値の5営業日変化（%）",
    "過去20日騰落率_%": "終値の20営業日変化（%）",
    "BB下限距離_%": "終値÷BB下限−1（%）",
    "BB幅_%": "（BB上限−BB下限）÷BB中央（%）",
    "ATR_%": "ATR÷終値（%）",
    "陽線実体_%": "（終値−始値）÷始値（%）。陰線は負",
    "出来高比_倍": "当日出来高÷直前20営業日の平均出来高",
    "確定高値の向き": "確認済み高値の切り上げ=1、切り下げ=-1、同値または2点未確定=0",
    "確定安値の向き": "確認済み安値の切り上げ=1、切り下げ=-1、同値または2点未確定=0",
    "前日米国4指数平均_%": "判定日より前の米国取引日の20営業日騰落率平均",
    "前日所属セクター_%": "判定日より前の米国取引日の所属セクターETF20営業日騰落率",
    "前日セクター順位": "同じ日の11セクター中の順位。1が最強",
    "前日セクターSPY差_pp": "所属セクター騰落率−SPY騰落率（pp）",
}


def causal_wave_direction(prices, width=3):
    """Only publish a pivot at i+width; never rewrite past feature rows."""
    pivots = []
    rows = []
    for t in range(len(prices)):
        i = t-width
        if i >= width:
            r = prices.iloc[i]
            neighbors = prices.iloc[i-width:i+width+1].drop(prices.index[i])
            high = bool(r.High > neighbors.High.max())
            low = bool(r.Low < neighbors.Low.min())
            if high != low:
                kind, price = ("高値", float(r.High)) if high else ("安値", float(r.Low))
                if pivots and pivots[-1][0] == kind:
                    if (high and price > pivots[-1][1]) or (low and price < pivots[-1][1]):
                        pivots[-1] = (kind, price)
                else:
                    pivots.append((kind, price))
        values = {}
        for kind, name in [("高値", "確定高値の向き"), ("安値", "確定安値の向き")]:
            points = [v for k,v in pivots if k == kind]
            values[name] = int(np.sign(points[-1]-points[-2])) if len(points) >= 2 else 0
        rows.append(values)
    return pd.DataFrame(rows,index=prices.index)


def build_stock_features(prices, atr_period=14, wave_width=3):
    p = prices.sort_index().copy()
    close = p.Close
    ma50, ma200 = close.rolling(50).mean(),close.rolling(200).mean()
    middle = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=0)
    lower,upper = middle-2*sd,middle+2*sd
    previous = close.shift(1)
    tr = pd.concat([p.High-p.Low,(p.High-previous).abs(),(p.Low-previous).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period,adjust=False,min_periods=atr_period).mean()
    x = pd.DataFrame(index=p.index)
    x["50日線乖離_%"] = (close/ma50-1)*100
    x["200日線乖離_%"] = (close/ma200-1)*100
    x["50日線の5日変化_%"] = (ma50/ma50.shift(5)-1)*100
    x["200日線の20日変化_%"] = (ma200/ma200.shift(20)-1)*100
    for h in [1,5,20]:x[f"過去{h}日騰落率_%"] = (close/close.shift(h)-1)*100
    x["BB下限距離_%"] = (close/lower-1)*100
    x["BB幅_%"] = (upper-lower)/middle*100
    x["ATR_%"] = atr/close*100
    x["陽線実体_%"] = (close/p.Open-1)*100
    if "Volume" in p and p.Volume.gt(0).any():
        x["出来高比_倍"] = p.Volume/p.Volume.shift(1).rolling(20).mean().replace(0,np.nan)
    x = x.join(causal_wave_direction(p,wave_width))
    return x.replace([np.inf,-np.inf],np.nan)


def align_past_context(features, context):
    """US context must be strictly older than stock day (safe across timezones)."""
    if context is None or context.empty:
        return features.copy()
    left = features.reset_index(names="日付")
    right = context.sort_index().reset_index(names="市場データ日")
    merged = pd.merge_asof(left.sort_values("日付"),right,left_on="日付",right_on="市場データ日",
        direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta(days=5))
    return merged.drop(columns="市場データ日").set_index("日付")


def build_targets(prices,horizon,threshold):
    future = (prices.Close.shift(-horizon)/prices.Close-1)*100
    known = pd.Series(prices.index,index=prices.index).shift(-horizon)
    labels = pd.Series(pd.NA,index=prices.index,dtype="object")
    valid = future.notna() & np.isfinite(future)
    labels.loc[valid] = "横ばい"
    labels.loc[valid & future.ge(threshold)] = "上昇"
    labels.loc[valid & future.le(-threshold)] = "下降"
    return pd.DataFrame({"結果":labels,"将来騰落率_%":future,"結果確定日":known})


def analyze(prices,start,end,horizon=5,threshold=2.0,atr_period=14,wave_width=3,context=None):
    if int(horizon)!=horizon or horizon<1 or threshold<=0:
        return {"error":"判定営業日数は1以上、上昇・下降の幅は0より大きく指定してください。"}
    p = prices.sort_index().loc[:pd.Timestamp(end)].copy()
    if p.empty:return {"error":"AI分析用の価格データがありません。"}
    x = build_stock_features(p,atr_period,wave_width)
    x = align_past_context(x,context)
    x = x.loc[(x.index>=pd.Timestamp(start)) & (x.index<=pd.Timestamp(end))]
    targets = build_targets(p,int(horizon),float(threshold)).reindex(x.index)
    core = [col for col in x if not col.startswith("前日")]
    base = x[core].dropna()
    notes = []
    if len(base)<180:
        return {"error":f"特徴が揃う日が{len(base)}日です。AI検証には少なくとも180日程度必要です。分析開始日を早めて再分析してください。", "features":x,"targets":targets}
    # Schema decisions are based on earlier data only, not validation labels.
    provisional_rows = base.join(targets).dropna(subset=["結果","結果確定日"])
    if len(provisional_rows)<170:
        return {"error":f"結果まで確定した分析可能日が{len(provisional_rows)}日です。期間を広げてください。", "features":x,"targets":targets}
    provisional = provisional_rows.index[int(len(provisional_rows)*.75)]
    schema_days = provisional_rows.index[(provisional_rows.index<provisional) & (provisional_rows["結果確定日"]<provisional)]
    selected = core.copy()
    for col in x:
        if col.startswith("前日"):
            coverage = x.loc[schema_days,col].notna().mean()
            if coverage>=.95:selected.append(col)
            else:notes.append(f"{col}：学習候補期間のデータ不足のため使用しません。")
    x = x[selected].dropna()
    labeled = x.join(targets).dropna(subset=["結果","結果確定日"])
    if len(labeled)<170:
        return {"error":f"結果まで確定した分析可能日が{len(labeled)}日です。期間を広げてください。", "features":x,"targets":targets}
    # Keep the boundary fixed before optional-feature coverage filtering.
    test_start = provisional
    # Purge every training label whose outcome is not yet known at test start.
    train = labeled[(labeled.index<test_start) & (labeled["結果確定日"]<test_start)]
    test = labeled[labeled.index>=test_start]
    if len(train)<120 or len(test)<40:
        return {"error":f"未来データ除外後：学習{len(train)}日・検証{len(test)}日。学習120日・検証40日以上となるよう期間を広げてください。", "features":x,"targets":targets}
    counts = train["結果"].value_counts()
    if len(counts)<2 or counts.min()<10:
        return {"error":"学習期間の上昇・下降・横ばいが偏っています。2分類以上で各10例以上必要です。期間を広げるか、判定幅を変更してください。", "features":x,"targets":targets}
    model = DecisionTreeClassifier(max_depth=3,min_samples_leaf=max(10,int(len(train)*.05)),random_state=42)
    model.fit(train[selected],train["結果"])
    pred = model.predict(test[selected])
    majority = counts.index[0]
    baseline = np.repeat(majority,len(test))
    accuracy = accuracy_score(test["結果"],pred)
    base_accuracy = accuracy_score(test["結果"],baseline)
    metrics = {"AI正解率":accuracy,"常に学習多数派を答える正解率":base_accuracy,
        "正解率差":accuracy-base_accuracy,"分類を均等に見た正解率":balanced_accuracy_score(test["結果"],pred),
        "Macro_F1":f1_score(test["結果"],pred,labels=CLASSES,average="macro",zero_division=0),
        "学習日数":len(train),"検証日数":len(test),"学習開始":train.index[0],"学習最終特徴日":train.index[-1],
        "学習最終結果確定日":train["結果確定日"].max(),"検証開始":test.index[0],"検証終了":test.index[-1],
        "除外した境界日数":int(((labeled.index<test_start)&(labeled["結果確定日"]>=test_start)).sum())}
    test = test.copy()
    test["AI判定"] = pred
    test["一致"] = test["結果"].eq(test["AI判定"])
    importance = permutation_importance(model,test[selected],test["結果"],scoring="accuracy",n_repeats=8,random_state=42,n_jobs=1)
    importance = pd.DataFrame({"特徴":selected,"正解率低下_pp":importance.importances_mean*100,"ばらつき_pp":importance.importances_std*100}).sort_values("正解率低下_pp",ascending=False)
    comparison = labeled.groupby("結果")[selected].mean().reindex(CLASSES)
    return {"error":None,"model":model,"features":x,"targets":targets,"train":train,"test":test,
        "metrics":metrics,"importance":importance,"comparison":comparison,"columns":selected,"notes":notes,
        "confusion":pd.DataFrame(confusion_matrix(test["結果"],pred,labels=CLASSES),index=CLASSES,columns=CLASSES),
        "horizon":int(horizon),"threshold":float(threshold),"test_start":test_start}


def explain_day(result,day):
    model,x = result["model"],result["features"]
    day = pd.Timestamp(day)
    if day<result["test_start"]:raise ValueError("説明対象は学習後の日付にしてください。")
    values = x.loc[[day],result["columns"]]
    leaf = int(model.apply(values)[0])
    predicted = str(model.predict(values)[0])
    rules = []
    node = 0
    visited = set(model.decision_path(values).indices)
    while model.tree_.feature[node]>=0:
        feature = result["columns"][model.tree_.feature[node]]
        threshold = float(model.tree_.threshold[node])
        value = float(values.iloc[0][feature])
        left = int(model.tree_.children_left[node]) in visited
        rules.append({"順番":len(rules)+1,"確認した特徴":feature,"当日の値":value,"境界値":threshold,
            "条件":f"{feature} {'≤' if left else '>'} {threshold:.4f}","特徴の意味":DEFINITIONS.get(feature,"")})
        node = model.tree_.children_left[node] if left else model.tree_.children_right[node]
    train = result["train"]
    similar = train.loc[model.apply(train[result["columns"]])==leaf].copy()
    # Training records in this exact tree leaf are the actual basis of its probabilities.
    distribution = similar["結果"].value_counts().reindex(CLASSES,fill_value=0)
    display = pd.DataFrame({"分類":CLASSES,"該当学習日数":distribution.values,
        "同じ条件での割合_%":distribution.values/len(similar)*100})
    actual = result["targets"].loc[day]
    return {"prediction":predicted,"rules":pd.DataFrame(rules),"distribution":display,
        "similar":similar,"actual":actual,"values":values,"leaf":leaf}
