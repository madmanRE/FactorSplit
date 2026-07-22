import datetime
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def shapley_decomposition(
        df: pd.DataFrame,
        date_col: str,
        segments: List[str],
        before: Tuple[datetime.date, datetime.date],
        after: Tuple[datetime.date, datetime.date],
        min_share_of_total: float = 0.01,
    ) -> Tuple[str, str]:

    before_0 = np.datetime64(before[0])
    before_1 = np.datetime64(before[1])
    after_0 = np.datetime64(after[0])
    after_1 = np.datetime64(after[1])

    before_df = df[(df[date_col] >= before_0) & (df[date_col] <= before_1)]
    after_df = df[(df[date_col] >= after_0) & (df[date_col] <= after_1)]

    total_before = before_df["clicks"].sum()

    before_agg = before_df.groupby(segments).agg(
        clicks_before=("clicks", "sum"),
        impressions_before=("impressions", "sum"),
        position_before=("position", "mean"),
    )
    before_agg["ctr_before"] = before_agg["clicks_before"] / before_agg["impressions_before"]

    after_agg = after_df.groupby(segments).agg(
        clicks_after=("clicks", "sum"),
        impressions_after=("impressions", "sum"),
        position_after=("position", "mean"),
    )
    after_agg["ctr_after"] = after_agg["clicks_after"] / after_agg["impressions_after"]

    merged = before_agg.join(after_agg, how="outer").fillna(0)
    merged["diff"] = merged["clicks_after"] - merged["clicks_before"]
    merged["diff_pct"] = np.where(
        merged["clicks_before"] > 0,
        merged["diff"] / merged["clicks_before"],
        np.nan
    )
    merged["share_of_total_before"] = merged["clicks_before"] / total_before

    merged_significant = merged[merged["share_of_total_before"] >= min_share_of_total].copy()

    def shapley_decompose(row):
        imp0, imp1 = row["impressions_before"], row["impressions_after"]
        pos0, pos1 = row["position_before"], row["position_after"]
        ctr0, ctr1 = row["ctr_before"], row["ctr_after"]

        def clicks_calc(imp, ctr):
            return imp * ctr

        step0 = clicks_calc(imp0, ctr0)
        step1a = clicks_calc(imp1, ctr0)
        step2a = clicks_calc(imp1, ctr1)
        contrib_imp_a = step1a - step0
        contrib_ctr_a = step2a - step1a

        step1b = clicks_calc(imp0, ctr1)
        step2b = clicks_calc(imp1, ctr1)
        contrib_ctr_b = step1b - step0
        contrib_imp_b = step2b - step1b

        on_impressions = (contrib_imp_a + contrib_imp_b) / 2
        on_ctr = (contrib_ctr_a + contrib_ctr_b) / 2

        return pd.Series({
            "on_impressions": on_impressions,
            "on_ctr": on_ctr,
            "position_delta": pos1 - pos0,
            "position_delta_pct": (pos1 - pos0) / pos0 if pos0 > 0 else np.nan,
        })

    decomposed = merged_significant.apply(shapley_decompose, axis=1)
    merged_significant = pd.concat([merged_significant, decomposed], axis=1)

    merged_significant = merged_significant.sort_values(by="diff", ascending=True)
    merged_significant = merged_significant.reset_index()

    return merged_significant


def get_interpretation(row):
    diff = row['diff']
    on_imp = row['on_impressions']
    on_ctr = row['on_ctr']
    pos_delta = row['position_delta']

    if abs(on_imp) > abs(on_ctr):
        driver = "показы"
    else:
        driver = "CTR"

    if diff > 0:
        direction = "роста"
        action = "выросли"
    else:
        direction = "снижения"
        action = "снизились"

    interpretation = f"Клики {action} на {diff:+,} ({row['diff_pct']:+.1%}). "

    if driver == "CTR" and pos_delta < -0.5:
        interpretation += (
            f"Основной фактор {direction} — CTR (вклад: {on_ctr:+.0f} кликов). "
            f"Позиция улучшилась на {abs(pos_delta):.1f} пунктов. "
            f"Вероятная причина: выход в топ выдачи и рост заметности."
        )
    elif driver == "CTR" and abs(pos_delta) <= 0.5:
        interpretation += (
            f"Основной фактор {direction} — CTR (вклад: {on_ctr:+.0f} кликов) "
            f"при стабильной позиции. "
            f"Вероятная причина: изменение сниппета или заголовка."
        )
    elif driver == "показы" and diff > 0:
        interpretation += (
            f"Основной фактор роста — показы (вклад: {on_imp:+.0f} кликов). "
            f"Вероятная причина: сезонность или расширение семантического охвата."
        )
    elif driver == "показы" and diff < 0:
        interpretation += (
            f"Основной фактор снижения — показы (вклад: {on_imp:+.0f} кликов). "
            f"Вероятная причина: потеря позиций или падение поискового спроса."
        )
    elif driver == "CTR" and pos_delta > 0.5:
        interpretation += (
            f"CTR вырос (вклад: {on_ctr:+.0f} кликов) несмотря на ухудшение позиции на {pos_delta:+.1f}. "
            f"Возможно улучшение сниппетов для оставшихся показов."
        )
    else:
        interpretation += "Смешанное влияние факторов."

    if driver == "CTR" and abs(pos_delta) <= 0.5:
        interpretation += " Рекомендация: зафиксировать изменения сниппетов и масштабировать на другие сегменты."
    elif driver == "показы" and diff > 0:
        interpretation += " Рекомендация: проанализировать поисковые запросы на предмет новых трендов."
    elif diff < 0:
        interpretation += " Требуется проверка технических проблем и позиций конкурентов."

    return interpretation

def build_segment_label(row, segment_cols):
    return "-".join(str(row[col]) for col in segment_cols)


def plot_shapley_analysis(data, segment_cols):
    """
    segment_cols: список названий колонок, определяющих сегмент
    (например ['group'], ['group', 'device'], ['country', 'device', 'query_type'] и т.д.)
    """

    st.header("Декомпозиция Шепли: Анализ причин изменения трафика")

    data = data.copy()
    data['segment_label'] = data.apply(lambda r: build_segment_label(r, segment_cols), axis=1)

    total_clicks_before = data['clicks_before'].sum()
    total_clicks_after = data['clicks_after'].sum()
    total_change = total_clicks_after - total_clicks_before
    total_change_pct = (total_change / total_clicks_before) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Клики (период до → после)",
            f"{total_clicks_before:,} → {total_clicks_after:,}",
            f"{total_change:+,} ({total_change_pct:+.1f}%)"
        )

    with col2:
        growing = len(data[data['diff'] > 0])
        falling = len(data[data['diff'] < 0])
        st.metric("Сегментов растёт / падает", f"{growing} / {falling}")

    with col3:
        dominant_impressions = len(data[data['on_impressions'].abs() > data['on_ctr'].abs()])
        dominant_ctr = len(data[data['on_ctr'].abs() > data['on_impressions'].abs()])
        st.metric("Драйвер: Показы / CTR", f"{dominant_impressions} / {dominant_ctr}")

    with col4:
        avg_position_change = data['position_delta'].mean()
        st.metric("Среднее изменение позиции", f"{avg_position_change:+.2f}", delta=None)

    st.divider()

    st.subheader("Водопад изменений по сегментам")

    data_sorted = data.sort_values('diff', ascending=True)

    fig_waterfall = go.Figure()

    fig_waterfall.add_trace(go.Waterfall(
        name="Изменение кликов",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(data_sorted) + ["total"],
        x=["Было"] + data_sorted['segment_label'].tolist() + ["Стало"],
        y=[total_clicks_before] + data_sorted['diff'].tolist() + [total_clicks_after],
        text=[f"{total_clicks_before:,}"] +
             [f"{v:+,}" for v in data_sorted['diff']] +
             [f"{total_clicks_after:,}"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
    ))

    fig_waterfall.update_layout(
        showlegend=False,
        height=500,
        yaxis_title="Клики",
        margin=dict(t=30, b=30)
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.subheader("Вклад показов и CTR по сегментам")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig_scatter = go.Figure()

        normalized_size = np.abs(data['diff']) / data['diff'].abs().max() * 50 + 10

        fig_scatter.add_trace(go.Scatter(
            x=data['on_impressions'],
            y=data['on_ctr'],
            mode='markers+text',
            text=data['segment_label'],
            textposition='top center',
            marker=dict(
                size=normalized_size,
                color=data['diff'],
                colorscale=[(0, '#e74c3c'), (0.5, '#f39c12'), (1, '#2ecc71')],
                showscale=True,
                colorbar=dict(title="Изменение кликов"),
                line=dict(width=1, color='white')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Вклад показов: %{x:+,.0f}<br>"
                "Вклад CTR: %{y:+,.0f}<br>"
                "<extra></extra>"
            )
        ))

        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig_scatter.update_layout(
            xaxis_title="Вклад показов (клики)",
            yaxis_title="Вклад CTR (клики)",
            height=500,
            margin=dict(t=30, b=30)
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("Обозначения")
        st.markdown("- Правый верхний квадрант: рост показов и CTR")
        st.markdown("- Левый верхний квадрант: падение показов компенсируется ростом CTR")
        st.markdown("- Правый нижний квадрант: рост показов при падении CTR")
        st.markdown("- Левый нижний квадрант: падение показов и CTR")
        st.markdown("---")
        st.markdown("Размер маркера пропорционален абсолютному изменению кликов")
        st.markdown("Цвет: зелёный — рост, красный — падение")

    st.divider()

    st.subheader("Детальный анализ по сегментам")

    for idx, row in data.iterrows():
        status_icon = "▲" if row['diff'] > 0 else "▼"
        with st.expander(f"{status_icon} {row['segment_label']}: {row['diff']:+,} кликов ({row['diff_pct']:+.1%})"):
            col1, col2, col3 = st.columns(3)

            with col1:
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Bar(
                    x=['Период до', 'Период после'],
                    y=[row['clicks_before'], row['clicks_after']],
                    marker_color=['#95a5a6', '#2ecc71' if row['diff'] > 0 else '#e74c3c'],
                    text=[f"{row['clicks_before']:,}", f"{row['clicks_after']:,}"],
                    textposition='auto',
                ))
                fig_metrics.update_layout(title="Клики", height=250, showlegend=False, margin=dict(t=40, b=30))
                st.plotly_chart(fig_metrics, use_container_width=True)

            with col2:
                fig_contrib = go.Figure()
                factors = ['Показы', 'CTR']
                values = [row['on_impressions'], row['on_ctr']]
                fig_contrib.add_trace(go.Bar(
                    x=factors,
                    y=values,
                    marker_color=['#3498db' if v > 0 else '#e74c3c' for v in values],
                    text=[f"{v:+.0f}" for v in values],
                    textposition='auto',
                ))
                fig_contrib.update_layout(title="Вклад факторов", height=250, showlegend=False, margin=dict(t=40, b=30))
                st.plotly_chart(fig_contrib, use_container_width=True)

            with col3:
                fig_pos = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=row['position_after'],
                    delta={
                        'reference': row['position_before'],
                        "decreasing": {"color": "#2ecc71"},
                        "increasing": {"color": "#e74c3c"},
                    },
                    title={'text': "Средняя позиция"},
                ))
                fig_pos.update_layout(height=250, margin=dict(t=40, b=30))
                st.plotly_chart(fig_pos, use_container_width=True)

            st.info(get_interpretation(row))

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Показы", f"{row['impressions_before']:,} → {row['impressions_after']:,}")
            with col_m2:
                st.metric("CTR", f"{row['ctr_before']:.2%} → {row['ctr_after']:.2%}")
            with col_m3:
                st.metric("Позиция", f"{row['position_before']:.2f} → {row['position_after']:.2f}")
            with col_m4:
                st.metric("Доля трафика", f"{row['share_of_total_before']:.1%}")

    st.divider()

    st.subheader("Парето-анализ вклада сегментов")

    data_pareto = data.copy()
    data_pareto['abs_diff'] = data_pareto['diff'].abs()
    data_pareto = data_pareto.sort_values('abs_diff', ascending=False).reset_index(drop=True)

    total_abs_diff = data_pareto['abs_diff'].sum()
    data_pareto['cum_pct'] = data_pareto['abs_diff'].cumsum() / total_abs_diff * 100

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])

    fig_pareto.add_trace(
        go.Bar(
            x=data_pareto['segment_label'],
            y=data_pareto['diff'],
            marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in data_pareto['diff']],
            name="Изменение кликов",
            text=[f"{v:+,}" for v in data_pareto['diff']],
            textposition='outside',
        ),
        secondary_y=False,
    )

    fig_pareto.add_trace(
        go.Scatter(
            x=data_pareto['segment_label'],
            y=data_pareto['cum_pct'],
            mode='lines+markers',
            name="Накопленный вклад, %",
            line=dict(color='#34495e', width=2),
        ),
        secondary_y=True,
    )

    fig_pareto.add_hline(y=80, line_dash="dash", line_color="gray", opacity=0.6, secondary_y=True)

    fig_pareto.update_layout(
        height=500,
        margin=dict(t=30, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_tickangle=-45,
    )
    fig_pareto.update_yaxes(title_text="Изменение кликов", secondary_y=False)
    fig_pareto.update_yaxes(title_text="Накопленный вклад, %", secondary_y=True, range=[0, 105])

    st.plotly_chart(fig_pareto, use_container_width=True)

    n_80 = (data_pareto['cum_pct'] <= 80).sum() + 1
    st.caption(
        f"{n_80} из {len(data_pareto)} сегментов формируют 80% совокупного изменения кликов "
        f"по модулю. Приоритет анализа и действий — на этих сегментах."
    )

    st.divider()

    st.subheader("CTR против изменения позиции")

    fig_ctr_pos = go.Figure()

    normalized_size_2 = np.abs(data['diff']) / data['diff'].abs().max() * 50 + 10

    fig_ctr_pos.add_trace(go.Scatter(
        x=data['position_delta'] * -1,
        y=data['on_ctr'],
        mode='markers+text',
        text=data['segment_label'],
        textposition='top center',
        marker=dict(
            size=normalized_size_2,
            color=data['diff'],
            colorscale=[(0, '#e74c3c'), (0.5, '#f39c12'), (1, '#2ecc71')],
            showscale=True,
            colorbar=dict(title="Изменение кликов"),
            line=dict(width=1, color='white')
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Улучшение позиции: %{x:+.2f}<br>"
            "Вклад CTR: %{y:+,.0f}<br>"
            "<extra></extra>"
        )
    ))

    fig_ctr_pos.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_ctr_pos.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_ctr_pos.update_layout(
        xaxis_title="Улучшение позиции (положительное значение = позиция выросла в топ)",
        yaxis_title="Вклад CTR (клики)",
        height=500,
        margin=dict(t=30, b=30)
    )

    st.plotly_chart(fig_ctr_pos, use_container_width=True)
    st.caption(
        "Правый верхний квадрант: рост CTR объясняется улучшением позиции, сниппет не требует правок. "
        "Левый верхний квадрант: CTR вырос при ухудшении или стабильной позиции — вероятно сработал сниппет или заголовок. "
        "Правый нижний квадрант: позиция выросла, но CTR не поддержал — проверить релевантность сниппета новой позиции."
    )

    st.divider()

    st.subheader("Распределение изменений по сегментам")

    fig_hist = go.Figure()

    fig_hist.add_trace(go.Histogram(
        x=data['diff_pct'] * 100,
        nbinsx=min(30, max(10, len(data) // 2)),
        marker_color='#3498db',
    ))

    mean_diff_pct = (data['diff_pct'] * 100).mean()
    fig_hist.add_vline(
        x=mean_diff_pct,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Среднее: {mean_diff_pct:+.1f}%",
    )
    fig_hist.add_vline(x=0, line_dash="dot", line_color="gray")

    fig_hist.update_layout(
        xaxis_title="Изменение кликов, %",
        yaxis_title="Число сегментов",
        height=400,
        margin=dict(t=30, b=30),
        showlegend=False,
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    skew_note = (
        "Распределение смещено в сторону снижения — большинство сегментов теряют трафик, "
        "похоже на системную причину (алгоритмическое обновление, сезонность, техническая проблема сайта)."
        if mean_diff_pct < -1 else
        "Распределение смещено в сторону роста — большинство сегментов растут, "
        "вероятна системная позитивная причина (сезонность спроса, снятие санкций, техническое улучшение)."
        if mean_diff_pct > 1 else
        "Распределение сбалансировано вокруг нуля — вероятно точечные, а не системные изменения по отдельным сегментам."
    )
    st.caption(skew_note)

    st.divider()

    st.subheader("Изменение кликов против доли трафика до периода")

    fig_share = go.Figure()

    normalized_size_3 = np.abs(data['diff']) / data['diff'].abs().max() * 50 + 10

    fig_share.add_trace(go.Scatter(
        x=data['share_of_total_before'] * 100,
        y=data['diff_pct'] * 100,
        mode='markers+text',
        text=data['segment_label'],
        textposition='top center',
        marker=dict(
            size=normalized_size_3,
            color=data['diff'],
            colorscale=[(0, '#e74c3c'), (0.5, '#f39c12'), (1, '#2ecc71')],
            showscale=True,
            colorbar=dict(title="Изменение кликов"),
            line=dict(width=1, color='white')
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Доля трафика до: %{x:.1f}%<br>"
            "Изменение: %{y:+.1f}%<br>"
            "<extra></extra>"
        )
    ))

    fig_share.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_share.update_layout(
        xaxis_title="Доля от общего трафика до периода, %",
        yaxis_title="Изменение кликов, %",
        height=500,
        margin=dict(t=30, b=30)
    )

    st.plotly_chart(fig_share, use_container_width=True)
    st.caption(
        "Сегменты в правой части графика формируют основную массу трафика — изменения там приоритетны "
        "даже при небольшом процентном отклонении. Сегменты слева могут показывать резкие процентные скачки, "
        "но малую долю в абсолютных кликах."
    )

    st.subheader("Сводная таблица")

    display_df = data[
        segment_cols + [
            'clicks_before', 'clicks_after', 'diff', 'diff_pct',
            'on_impressions', 'on_ctr', 'position_delta'
        ]
    ].copy()

    display_df['diff_pct'] = display_df['diff_pct'].apply(lambda x: f"{x:+.1%}")
    display_df['position_delta'] = display_df['position_delta'].apply(lambda x: f"{x:+.2f}")

    def color_values(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #2ecc71'
            elif val < 0:
                return 'color: #e74c3c'
        return ''

    st.dataframe(
        display_df.style.map(color_values, subset=['diff', 'on_impressions', 'on_ctr']),
        use_container_width=True,
        hide_index=True
    )
