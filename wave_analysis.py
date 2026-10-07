"""Descriptive analysis of hindsight-defined, completed price waves. No forecast API."""
import numpy as np
import pandas as pd

FEATURES = ["過去1日騰落率_%", "過去5日騰落率_%", "過去20日騰落率_%",
    "50日線乖離_%", "200日線乖離_%", "50日線の5日変化_%", "200日線の20日変化_%",
    "出来高比_倍", "BB内位置_%", "BB幅_%", "ATR_%", "実体_%", "BB下限接触", "BB上限接触"]
DIRECTIONS = ["安値→高値", "高値→安値"]
NOTE = ("表示チャートと同じ事後確定の転換点を使用。開始条件は転換点当日の終値時点です。"
    "転換点の判定には後日の足を使い、同種の点は後からより極端な点に置き換わる場合があります。"
    "値幅は安値・高値間の理論的な変化で、実現できる売買利益ではありません。"
    "条件との関連を示す結果分析であり、原因・必要条件・十分条件や将来予測ではありません。")

def completed_waves(path, swings):
    x = path.copy()
    x["日付"] = pd.to_datetime(x["日付"])
    x = x.sort_values("日付").reset_index(drop=True)
    if x["日付"].duplicated().any(): raise ValueError("株価の日付が重複しています。")
    lookup = {d:i for i,d in enumerate(x["日付"])}
    rows = []
    pivots = swings.sort_values("日付").to_dict("records") if not swings.empty else []
    for a,b in zip(pivots,pivots[1:]):
        ad,bd = pd.Timestamp(a["日付"]),pd.Timestamp(b["日付"])
        if a["種類"] == b["種類"] or ad not in lookup or bd not in lookup: continue
        ap,bp = float(a["価格"]),float(b["価格"])
        if ap <= 0 or not np.isfinite([ap,bp]).all() or bd <= ad: continue
        move = (bp/ap-1)*100
        r = {"波番号":len(rows)+1,"方向":"安値→高値" if a["種類"]=="安値" else "高値→安値",
            "開始日":ad,"終了日":bd,"開始確定日":pd.Timestamp(a["確定日"]),
            "終了確定日":pd.Timestamp(b["確定日"]),"開始価格":ap,"終了価格":bp,
            "営業日数":lookup[bd]-lookup[ad],"変化_%":move,"絶対値幅_%":abs(move)}
        snap=x.iloc[lookup[ad]]
        for f in FEATURES:
            r[f]=pd.to_numeric(pd.Series([snap.get(f,np.nan)]),errors="coerce").iloc[0]
        rows.append(r)
    return pd.DataFrame(rows,columns=["波番号","方向","開始日","終了日","開始確定日","終了確定日",
        "開始価格","終了価格","営業日数","変化_%","絶対値幅_%"]+FEATURES)

def feature_comparison(waves):
    rows=[]
    for f in FEATURES:
        r={"開始日の条件":f}
        for direction in DIRECTIONS:
            values=waves.loc[waves["方向"].eq(direction),f].replace([np.inf,-np.inf],np.nan).dropna()
            r[direction+" 有効波数"]=len(values)
            r[direction+" 中央値"]=values.median() if len(values) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def condition_rates(waves):
    rules=[("50日線より下","50日線乖離_%",lambda x:x<0),
        ("200日線より下","200日線乖離_%",lambda x:x<0),
        ("50日線が上向き","50日線の5日変化_%",lambda x:x>0),
        ("200日線が上向き","200日線の20日変化_%",lambda x:x>0),
        ("直前5日が下落","過去5日騰落率_%",lambda x:x<0),
        ("出来高が直前20日平均の1.5倍以上","出来高比_倍",lambda x:x>=1.5),
        ("当日の安値がBB下限以下","BB下限接触",lambda x:x==1),
        ("当日の高値がBB上限以上","BB上限接触",lambda x:x==1)]
    rows=[]
    for label,f,test in rules:
        for direction in DIRECTIONS:
            v=waves.loc[waves["方向"].eq(direction),f].replace([np.inf,-np.inf],np.nan).dropna()
            n=int(test(v).sum())
            rows.append({"条件":label,"方向":direction,"該当波数":n,"確認可能波数":len(v),
                "該当割合_%":n/len(v)*100 if len(v) else np.nan})
    return pd.DataFrame(rows)

def historical_groups(waves, direction):
    """Fit only completed historical waves, explaining observed amplitude differences."""
    from sklearn.tree import DecisionTreeRegressor
    w=waves.loc[waves["方向"].eq(direction)].copy()
    # Omit wholly absent or constant inputs; never replace missing conditions with zero.
    fs=[f for f in FEATURES if w[f].replace([np.inf,-np.inf],np.nan).nunique()>1]
    w=w.replace([np.inf,-np.inf],np.nan).dropna(subset=fs+["絶対値幅_%"])
    if len(w)<12 or not fs or w["絶対値幅_%"].nunique()<2:
        return pd.DataFrame(),f"利用可能な波は{len(w)}件。条件別整理には同方向の波12件以上、変動する条件と値幅が必要です。"
    model=DecisionTreeRegressor(max_depth=2,min_samples_leaf=3,random_state=0)
    model.fit(w[fs],w["絶対値幅_%"])
    leaves=model.apply(w[fs]); rows=[]; tree=model.tree_
    def visit(node,conditions):
        if tree.children_left[node]==tree.children_right[node]:
            z=w.loc[leaves==node]
            rows.append({"条件グループ":len(rows)+1,"開始日の条件の組合せ":" かつ ".join(conditions) or "分割なし",
                "波数":len(z),"値幅中央値_%":z["絶対値幅_%"].median(),
                "値幅平均_%":z["絶対値幅_%"].mean(),"値幅最小_%":z["絶対値幅_%"].min(),
                "値幅最大_%":z["絶対値幅_%"].max(),"該当波番号":", ".join(z["波番号"].astype(str))})
            return
        f=fs[tree.feature[node]]; t=tree.threshold[node]
        visit(tree.children_left[node],conditions+[f"{f} ≤ {t:.6g}"])
        visit(tree.children_right[node],conditions+[f"{f} > {t:.6g}"])
    visit(0,[])
    return pd.DataFrame(rows),f"同方向の{len(w)}波を使用（欠損のある波は除外）。最大2段、各グループ最低3波。条件表示は丸め値です。"
