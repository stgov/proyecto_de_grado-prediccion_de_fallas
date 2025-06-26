import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

from utils.data_loader import cargar_raw, cargar_process_robot, cargar_avg_global

st.set_page_config(
    page_title="Dashboard Analisis RPA",
    page_icon="🤖",
    layout="wide",
)

st.title("Análisis Exploratorio de Datos RPA")

default_page_length_options = [10, 25, 50, 100, 200]

main_col1, main_col2 = st.columns(2)

with main_col1:
    st.header("Vista previa datos")
    st.caption("Datos semi procesados de RPA")

    df_raw = cargar_raw()

    default_page_length_raw = 50

    selected_page_length_raw = st.session_state.get("page_length_raw", default_page_length_raw)
    current_page_raw = st.session_state.get("current_page_raw", 1)

    total_rows_raw = len(df_raw)
    num_pages_raw = (total_rows_raw + selected_page_length_raw - 1) // selected_page_length_raw if selected_page_length_raw > 0 else 1
    page_options_raw = list(range(1, num_pages_raw + 1)) if num_pages_raw > 0 else [1]

    if current_page_raw > num_pages_raw:
        current_page_raw = num_pages_raw
    elif current_page_raw < 1:
        current_page_raw = 1

    start_row_raw = (current_page_raw - 1) * selected_page_length_raw
    end_row_raw = start_row_raw + selected_page_length_raw

    st.dataframe(df_raw.iloc[start_row_raw:end_row_raw], use_container_width=True)

    col1_raw_controls, col2_raw_controls, col3_raw_controls = st.columns([3, 1, 1])

    with col2_raw_controls:
        st.selectbox(
            "Filas por página:",
            options=default_page_length_options,
            index=default_page_length_options.index(selected_page_length_raw),
            key="page_length_raw"
        )

    with col3_raw_controls:
        st.selectbox(
            "Página:",
            options=page_options_raw,
            index=page_options_raw.index(current_page_raw) if current_page_raw in page_options_raw else 0,
            key="current_page_raw"
        )

with main_col2:
    st.header("Datos Agrupados")
    st.caption("Por 'process_name' y 'user_robot' por día")

    df_process_robot = cargar_process_robot()

    default_page_length_proc = 50

    selected_page_length_proc = st.session_state.get("page_length_proc", default_page_length_proc)
    current_page_proc = st.session_state.get("current_page_proc", 1)

    total_rows_proc = len(df_process_robot)
    num_pages_proc = (total_rows_proc + selected_page_length_proc - 1) // selected_page_length_proc if selected_page_length_proc > 0 else 1
    page_options_proc = list(range(1, num_pages_proc + 1)) if num_pages_proc > 0 else [1]

    if current_page_proc > num_pages_proc:
        current_page_proc = num_pages_proc
    elif current_page_proc < 1:
        current_page_proc = 1

    start_row_proc = (current_page_proc - 1) * selected_page_length_proc
    end_row_proc = start_row_proc + selected_page_length_proc

    st.dataframe(df_process_robot.iloc[start_row_proc:end_row_proc], use_container_width=True)

    col1_proc_controls, col2_proc_controls, col3_proc_controls = st.columns([3, 1, 1])

    with col2_proc_controls:
        st.selectbox(
            "Filas por página:",
            options=default_page_length_options,
            index=default_page_length_options.index(selected_page_length_proc),
            key="page_length_proc"
        )

    with col3_proc_controls:
        st.selectbox(
            "Página:",
            options=page_options_proc,
            index=page_options_proc.index(current_page_proc) if current_page_proc in page_options_proc else 0,
            key="current_page_proc"
        )

st.markdown("---")

st.header("Análisis Gráfico de Series de Tiempo")

plot_col1, plot_col2 = st.columns(2)

df_avg_global = cargar_avg_global()

# Calcular las columnas 'Completed' y 'Exception' a partir de 'entries' y 'tasa_excepcion'
# Dado que 'tasa_excepcion' va entre 0 y 1
df_avg_global['Exception'] = df_avg_global['entries'] * df_avg_global['tasa_excepcion']
df_avg_global['Completed'] = df_avg_global['entries'] - df_avg_global['Exception']

with plot_col1:
    st.subheader("Entradas por Día (Completed vs. Exception)")

    fig_entradas = go.Figure(data=[
        go.Bar(name='Completed', x=df_avg_global.index, y=df_avg_global['Completed'], marker_color='blue'),
        go.Bar(name='Exception', x=df_avg_global.index, y=df_avg_global['Exception'], marker_color='red')
    ])
    fig_entradas.update_layout(barmode='stack', title_text='Número de Entradas Diarias',
                              xaxis_title="Fecha", yaxis_title="Número de Entradas")
    st.plotly_chart(fig_entradas, use_container_width=True)

with plot_col2:
    st.subheader("Tasa de Excepción a Través del Tiempo")

    fig_tasa_excepcion = go.Figure(data=[
        go.Scatter(name='Tasa de Excepción', x=df_avg_global.index, y=df_avg_global['tasa_excepcion'],
                   mode='lines+markers', marker_color='purple')
    ])
    fig_tasa_excepcion.update_layout(title_text='Tasa de Excepción Diaria',
                                    xaxis_title="Fecha", yaxis_title="Tasa de Excepción")
    st.plotly_chart(fig_tasa_excepcion, use_container_width=True)