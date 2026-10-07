import pandas as pd
import streamlit as st
from wave_analysis import (FEATURES, DIRECTIONS, NOTE, completed_waves,
    feature_comparison, condition_rates, historical_groups)

def show_wave_analysis(result, path, swings, width):
    st.markdown("#### 過去の波の結果分析：どの条件が揃っていたか")
    st.caption(f"{result['ticker']}｜上のチャートと同じ左右{width}本の転換点を使用。銘柄・期間・確認本数を変えると再集計します。")
    st.info(NOTE)
    waves=completed_waves(path,swings)
    if waves.empty:
        st.info("完了した波がありません。転換点が2点以上ある期間を選んでください。")
        return
    st.caption("隣り合う確定転換点を1つの波として集計します。最後の転換点以降の未完了の波は含みません。")
    cols=st.columns(2)
    for col,d in zip(cols,DIRECTIONS):
        z=waves[ waves["方向"].eq(d)]
        with col:
            st.metric(d+" 完了波数",f"{len(z)}波")
            st.write(f"値幅の絶対値の中央値：{z['絶対値幅_%'].median():.2f}% ／営業日数の中央値：{z['営業日数'].median():.1f}日" if len(z) else "集計対象なし")
    overview=waves.drop(columns=FEATURES)
    st.markdown("**波の一覧：いつからいつまで、どれだけ動いたか**")
    st.dataframe(overview,use_container_width=True,hide_index=True)
    st.download_button("開始条件付きの全波CSVを保存",waves.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{result['ticker']}_historical_waves.csv",mime="text/csv",key="wave_csv")
    selected=st.selectbox("詳しく確認する過去の波",waves["波番号"].tolist(),
        format_func=lambda n: f"波{n}｜{waves.loc[waves['波番号'].eq(n),'方向'].iloc[0]}｜{waves.loc[waves['波番号'].eq(n),'開始日'].iloc[0]:%Y-%m-%d}",key="wave_detail")
    r=waves.loc[waves["波番号"].eq(selected)].iloc[0]
    st.write(f"{r['開始日']:%Y-%m-%d} → {r['終了日']:%Y-%m-%d}：変化 {r['変化_%']:+.2f}%、{int(r['営業日数'])}営業日")
    st.caption(f"開始転換点の確定日 {r['開始確定日']:%Y-%m-%d} ／終了転換点の確定日 {r['終了確定日']:%Y-%m-%d}")
    st.dataframe(pd.DataFrame({"開始日に確認した項目":FEATURES,"値":[r[f] for f in FEATURES]}),use_container_width=True,hide_index=True)
    st.caption("BB内位置：下限0・上限100（範囲外もあります）。BB接触：1＝接触、0＝非接触。出来高比：当日÷直前20営業日の平均。欠損は未確認で、0ではありません。実体は終値÷始値−1。")
    comparison=feature_comparison(waves); rates=condition_rates(waves)
    st.markdown("**上昇の波・下降の波で、開始日の状態はどう違ったか**")
    st.dataframe(comparison,use_container_width=True,hide_index=True)
    st.markdown("**確認条件を忘れないための一覧**")
    st.dataframe(rates,use_container_width=True,hide_index=True)
    st.caption("割合は、その方向の波のうち条件が揃った割合です。条件が揃った日に上昇する確率ではありません。BB接触項目の中央値は0/1の中央値です。")
    st.markdown("**AIによる過去の条件の組合せ整理**")
    st.caption("過去に完了した波だけを使い、開始条件で値幅が異なったグループを決定木で整理します。全件を整理用に学習するため、予測成績ではありません。別期間・別銘柄で同じ関連が出るかを確認するための探索結果です。")
    groups=[]
    for direction in DIRECTIONS:
        table,note=historical_groups(waves,direction)
        st.markdown(f"**{direction}**")
        st.caption(note)
        if not table.empty:
            st.dataframe(table,use_container_width=True,hide_index=True)
            groups.append(direction+"\n"+table.to_csv(index=False))
    report=(f"過去の波の結果分析レポート v3.0.0\n銘柄: {result['ticker']}\n"
        f"期間: {result['analysis_start']} ～ {result['analysis_end']}\n転換点: 左右{width}本\n"
        "過去の波の開始条件と値幅の関連を確認してください。将来予測ではありません。\n"
        "【波と開始条件】\n"+waves.to_csv(index=False)+"\n【条件の比較】\n"+comparison.to_csv(index=False)
        +"\n【条件の該当割合】\n"+rates.to_csv(index=False)+"\n【AIによる整理】\n"+"\n".join(groups)+"\n【注意】\n"+NOTE)
    with st.expander("結果を相談するためのレポート"):
        st.caption("右上のコピーアイコンから全文をコピーできます。")
        st.code(report,language="text")
        st.download_button("相談用レポートを保存",report.encode("utf-8-sig"),
            file_name=f"{result['ticker']}_wave_report.txt",mime="text/plain",key="wave_report")
