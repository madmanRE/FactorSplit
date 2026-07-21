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


def plot_shapley_analysis(data):
    st.header("Декомпозиция Шепли: Анализ причин изменения трафика")

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
        st.metric(
            "Среднее изменение позиции",
            f"{avg_position_change:+.2f}",
            delta=None
        )

    st.divider()

    st.subheader("Водопад изменений по сегментам")

    data_sorted = data.sort_values('diff', ascending=True)

    fig_waterfall = go.Figure()

    fig_waterfall.add_trace(go.Waterfall(
        name="Изменение кликов",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(data_sorted) + ["total"],
        x=["Было"] + [f"{row['group']}-{row['device']}" for _, row in data_sorted.iterrows()] + ["Стало"],
        y=[total_clicks_before] + data_sorted['diff'].tolist() + [total_clicks_after],
        text=[f"{total_clicks_before:,}"] +
             [f"{row['diff']:+,}" for _, row in data_sorted.iterrows()] +
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
            text=data.apply(lambda x: f"{x['group']}-{x['device']}", axis=1),
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
        st.markdown("**Обозначения**")
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
        with st.expander(
                f"{status_icon} {row['group']}-{row['device']}: {row['diff']:+,} кликов ({row['diff_pct']:+.1%})"):
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

                fig_metrics.update_layout(
                    title="Клики",
                    height=250,
                    showlegend=False,
                    margin=dict(t=40, b=30)
                )

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

                fig_contrib.update_layout(
                    title="Вклад факторов",
                    height=250,
                    showlegend=False,
                    margin=dict(t=40, b=30)
                )

                st.plotly_chart(fig_contrib, use_container_width=True)

            with col3:
                fig_pos = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=row['position_after'],
                    delta={'reference': row['position_before']},
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

    st.subheader("Тепловая карта изменений")

    pivot_diff = data.pivot_table(
        values='diff_pct',
        index='group',
        columns='device',
        aggfunc='first'
    )

    pivot_ctr = data.pivot_table(
        values='on_ctr',
        index='group',
        columns='device',
        aggfunc='first'
    )

    fig_heatmap = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Изменение кликов (%)', 'Вклад CTR (клики)'),
        horizontal_spacing=0.15
    )

    fig_heatmap.add_trace(
        go.Heatmap(
            z=pivot_diff.values,
            x=pivot_diff.columns,
            y=pivot_diff.index,
            text=[[f"{v:+.1%}" for v in row] for row in pivot_diff.values],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=[(0, '#e74c3c'), (0.5, '#f39c12'), (1, '#2ecc71')],
            showscale=False,
            zmid=0
        ),
        row=1, col=1
    )

    fig_heatmap.add_trace(
        go.Heatmap(
            z=pivot_ctr.values,
            x=pivot_ctr.columns,
            y=pivot_ctr.index,
            text=[[f"{v:+.0f}" for v in row] for row in pivot_ctr.values],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=[(0, '#e74c3c'), (0.5, '#ecf0f1'), (1, '#2ecc71')],
            showscale=False,
            zmid=0
        ),
        row=1, col=2
    )

    fig_heatmap.update_layout(height=400, margin=dict(t=40, b=30))

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Сводная таблица")

    display_df = data[[
        'group', 'device',
        'clicks_before', 'clicks_after', 'diff', 'diff_pct',
        'on_impressions', 'on_ctr',
        'position_delta'
    ]].copy()

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


def get_interpretation(row):
    group = row['group']
    device = row['device']
    diff = row['diff']
    on_imp = row['on_impressions']
    on_ctr = row['on_ctr']
    pos_delta = row['position_delta']

    if abs(on_imp) > abs(on_ctr):
        driver = "показы"
        driver_value = on_imp
    else:
        driver = "CTR"
        driver_value = on_ctr

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