import os
import datetime
import streamlit as st
import pandas as pd
from core import shapley_decomposition, plot_shapley_analysis


st.write("""
# 📊 Factor Split
""")

with st.expander("📘 Инструкция по использованию"):
    st.markdown(
        """
        #### ❓ FAQ: Как работает программа

        **1. Пользователь отправляет файл**, который должен
        содержать дату, 1 или более колонок для сегментов (например country, device, search_engine),
        а также в файле должны быть колонки clicks, impressions, ctr, position

        **2. Пользователь указывает** название колонки с датой,
        названия колонок сегментов, выбирает 2 диапазона дат и 
        указывает параметр по которому будут фильтроваться сегменты (порог фиксации сегмента). 
        Например если значение порога фиксации сегмента = 0.01, то все
        сегменты, чье кол-во кликов менее 1% от общих кликов будут отфильтрованы.

        #### ❗ ВАЖНО:
        * Колонки clicks, impressions, ctr, position обязательно должны быть в файле
        * Самостоятельно требуется подготовить файл, а именно: обработать пропуски, заменить типы данные
        с текстовых на числовые для колонок clicks, impressions, ctr, position

        """
    )

with st.form("main_form"):
    file = st.file_uploader("Загрузите файл", type=["csv", "xlsx"])
    date_col = st.text_input("Введите название колонки с датами")
    segments = st.text_area("Укажите названия колонок для агрегации")
    before = st.date_input(
        "Выберете диапазон ДО",
        value=(datetime.date.today(), datetime.date.today() + datetime.timedelta(days=7)),
        format="YYYY/MM/DD"
    )
    after = st.date_input(
        "Выберете диапазон ПОСЛЕ",
        value=(datetime.date.today(), datetime.date.today() + datetime.timedelta(days=7)),
        format="YYYY/MM/DD"
    )
    min_share_of_total = st.slider(
        label="Выберете минимальный порог фиксации сегмента",
        min_value=0.0,
        max_value=0.1,
        value=0.01,
        step=0.01
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        file_name, file_extension = os.path.splitext(file.name)
        if file_extension == ".csv":
            df = pd.read_csv(file, parse_dates=[date_col])
        elif file_extension == ".xlsx":
            df = pd.read_excel(file, parse_dates=[date_col])
        else:
            st.error("Формат файла должен быть CSV или XLSX")

        segments = [s.strip() for s in segments.split("\n")]

        st.set_page_config(page_title="Shapley SEO Decomposition", layout="wide")
        st.title("Анализ причин изменения трафика")
        data = shapley_decomposition(df, date_col, segments, before, after, min_share_of_total)
        plot_shapley_analysis(data, segment_cols=segments)
