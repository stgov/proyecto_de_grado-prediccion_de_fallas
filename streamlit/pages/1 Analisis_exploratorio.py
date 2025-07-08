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
    # Limitar a máximo 3 páginas
    num_pages_raw = min(num_pages_raw, 3)
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
    # Limitar a máximo 3 páginas
    num_pages_proc = min(num_pages_proc, 3)
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

st.markdown("---")

st.header("Análisis de Período Específico")

# Controles para seleccionar el rango de fechas
date_col1, date_col2 = st.columns(2)
with date_col1:
    start_date = st.date_input("Fecha de inicio", df_avg_global.index.min())
with date_col2:
    end_date = st.date_input("Fecha de fin", df_avg_global.index.max())

# Convertir a Timestamp para la comparación
start_date_dt = pd.to_datetime(start_date).date()
end_date_dt = pd.to_datetime(end_date).date()

# Filtrar los datos para el período seleccionado
filtered_avg_global = df_avg_global.loc[start_date:end_date]

if 'fecha' not in df_process_robot.columns or df_process_robot['fecha'].dtype != 'object':
    # Asegurarse de que el índice sea DatetimeIndex antes de acceder a .date
    if not isinstance(df_process_robot.index, pd.DatetimeIndex):
        df_process_robot.index = pd.to_datetime(df_process_robot.index)
    df_process_robot['fecha'] = df_process_robot.index.date

filtered_process_robot = df_process_robot[
    (df_process_robot['fecha'] >= start_date_dt) &
    (df_process_robot['fecha'] <= end_date_dt)
]

if 'fecha' not in df_raw.columns or df_raw['fecha'].dtype != 'object':
    df_raw['fecha'] = pd.to_datetime(df_raw['Loaded']).dt.date
    
filtered_raw = df_raw[
    (df_raw['fecha'] >= start_date_dt) &
    (df_raw['fecha'] <= end_date_dt)
]


if not filtered_avg_global.empty:
    # Gráfico de tasa de excepción para el período
    st.subheader(f"Tasa de Excepción Diaria ({start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')})")
    fig_filtered_tasa = go.Figure(data=[
        go.Scatter(
            x=filtered_avg_global.index,
            y=filtered_avg_global['tasa_excepcion'],
            mode='lines+markers',
            marker_color='green'
        )
    ])
    fig_filtered_tasa.update_layout(
        title_text='Tasa de Excepción para el Período Seleccionado',
        xaxis_title="Fecha",
        yaxis_title="Tasa de Excepción"
    )
    st.plotly_chart(fig_filtered_tasa, use_container_width=True)

    # Análisis de fallos por robot y proceso
    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.subheader("Ranking de Robots con más Excepciones")
        if not filtered_process_robot.empty:
            exceptions_by_robot = filtered_process_robot.copy()
            exceptions_by_robot['exception_count'] = exceptions_by_robot['entries'] * exceptions_by_robot['tasa_excepcion']
            failed_robots = exceptions_by_robot.groupby('UserRobot')['exception_count'].sum().sort_values(ascending=False).reset_index()
            st.dataframe(failed_robots, use_container_width=True)

            if not failed_robots.empty:
                top_robot = failed_robots.iloc[0]['UserRobot']
                st.subheader(f"Razones de Excepción para {top_robot}")
                robot_exceptions = filtered_raw[
                    (filtered_raw['UserRobot'] == top_robot) &
                    (filtered_raw['FinishStatus'] == 'Exception')
                ]
                if not robot_exceptions.empty:
                    reason_counts = robot_exceptions['ExceptionReason'].value_counts().reset_index()
                    reason_counts.columns = ['Razón de Excepción', 'Cantidad']
                    st.table(reason_counts)
                else:
                    st.info(f"No se encontraron razones de excepción para el robot {top_robot}.")

        else:
            st.info("No hay datos de robots para el período seleccionado.")

    with analysis_col2:
        st.subheader("Ranking de Procesos con más Excepciones")
        if not filtered_process_robot.empty:
            exceptions_by_process = filtered_process_robot.copy()
            exceptions_by_process['exception_count'] = exceptions_by_process['entries'] * exceptions_by_process['tasa_excepcion']
            failed_processes = exceptions_by_process.groupby('ProcessName')['exception_count'].sum().sort_values(ascending=False).reset_index()
            st.dataframe(failed_processes, use_container_width=True)

            if not failed_processes.empty:
                top_process = failed_processes.iloc[0]['ProcessName']
                st.subheader(f"Razones de Excepción para {top_process}")
                process_exceptions = filtered_raw[
                    (filtered_raw['ProcessName'] == top_process) &
                    (filtered_raw['FinishStatus'] == 'Exception')
                ]
                if not process_exceptions.empty:
                    reason_counts = process_exceptions['ExceptionReason'].value_counts().reset_index()
                    reason_counts.columns = ['Razón de Excepción', 'Cantidad']
                    st.table(reason_counts)
                else:
                    st.info(f"No se encontraron razones de excepción para el proceso {top_process}.")
        else:
            st.info("No hay datos de procesos para el período seleccionado.")

    st.markdown("---")
    st.subheader("Análisis de Series de Tiempo Individuales")

    # Obtener robots y procesos únicos del período filtrado
    unique_robots = filtered_process_robot['UserRobot'].unique()
    unique_processes = filtered_process_robot['ProcessName'].unique()

    # Selectores para robots y procesos
    select_col1, select_col2 = st.columns(2)
    with select_col1:
        selected_robots = st.multiselect("Seleccionar Robots para visualizar su serie", options=unique_robots)
    with select_col2:
        selected_processes = st.multiselect("Seleccionar Procesos para visualizar su serie", options=unique_processes)

    # Gráficos para las series de tiempo individuales
    plot_series_col1, plot_series_col2 = st.columns(2)
    with plot_series_col1:
        if selected_robots:
            fig_robot_series = go.Figure()
            for robot in selected_robots:
                robot_data = filtered_process_robot[filtered_process_robot['UserRobot'] == robot].sort_index()
                fig_robot_series.add_trace(go.Scatter(x=robot_data.index, y=robot_data['tasa_excepcion'], name=robot, mode='lines+markers'))
            fig_robot_series.update_layout(title_text="Tasa de Excepción por Robot", xaxis_title="Fecha", yaxis_title="Tasa de Excepción")
            st.plotly_chart(fig_robot_series, use_container_width=True)
        else:
            st.info("Seleccione al menos un robot para ver su serie de tiempo.")

    with plot_series_col2:
        if selected_processes:
            fig_process_series = go.Figure()
            for process in selected_processes:
                process_data = filtered_process_robot[filtered_process_robot['ProcessName'] == process].sort_index()
                fig_process_series.add_trace(go.Scatter(x=process_data.index, y=process_data['tasa_excepcion'], name=process, mode='lines+markers'))
            fig_process_series.update_layout(title_text="Tasa de Excepción por Proceso", xaxis_title="Fecha", yaxis_title="Tasa de Excepción")
            st.plotly_chart(fig_process_series, use_container_width=True)
        else:
            st.info("Seleccione al menos un proceso para ver su serie de tiempo.")

else:
    st.warning("No hay datos disponibles para el rango de fechas seleccionado.")