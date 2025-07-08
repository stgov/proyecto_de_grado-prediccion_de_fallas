import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

st.set_page_config(
    page_title="Dashboard Analisis RPA",
    page_icon="🤖",
    layout="wide",
)

st.title("Predicción de Series de Tiempo")
st.write("Utiliza el modelo entrenado para realizar pronósticos futuros de la tasa de excepción global.")

# Verificar si el modelo está en el estado de la sesión
if 'model' not in st.session_state or 'model_choice' not in st.session_state:
    st.warning("El modelo no ha sido entrenado. Por favor, ve a la página 'Modelamiento' y entrena un modelo primero.")
    st.stop()

# Cargar modelo y datos desde el estado de la sesión
model = st.session_state['model']
model_choice = st.session_state['model_choice']
serie_a_modelar = st.session_state['serie_a_modelar']
train_data = st.session_state['train_data']

st.subheader(f"Configuración del Pronóstico Futuro con {model_choice}")

last_date_in_data = serie_a_modelar.index[-1]

forecast_horizon = st.slider(
    "Número de días a pronosticar hacia el futuro:",
    min_value=1,
    max_value=90,
    value=7,
    step=1
)

forecast_start_date = last_date_in_data + pd.Timedelta(days=1)
forecast_end_date = last_date_in_data + pd.Timedelta(days=forecast_horizon)

st.write(f"Pronosticando desde **{forecast_start_date.strftime('%Y-%m-%d')}** hasta **{forecast_end_date.strftime('%Y-%m-%d')}**.")

# Realizar el pronóstico según el modelo seleccionado
if model_choice == "ETS":
    future_forecast = model.predict(start=forecast_start_date, end=forecast_end_date)

elif model_choice == "TBATS":
    # El método forecast de SimpleTBATS devuelve valores y un intervalo de confianza (dummy)
    forecast_values, _ = model.forecast(steps=forecast_horizon)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_horizon, freq='D')
    future_forecast = pd.Series(forecast_values, index=forecast_index)

st.subheader("Visualización del Pronóstico")

fig_prediction = go.Figure()

fig_prediction.add_trace(go.Scatter(x=serie_a_modelar.index, y=serie_a_modelar, mode='lines', name='Datos Históricos', line=dict(color='blue')))

# Conectar los datos históricos con el pronóstico
last_historical_index = serie_a_modelar.index[-1]
last_historical_value = serie_a_modelar.iloc[-1]
extended_future_forecast_values = pd.concat([pd.Series([last_historical_value], index=[last_historical_index]), future_forecast])
extended_future_forecast_values = extended_future_forecast_values.loc[~extended_future_forecast_values.index.duplicated(keep='last')]

fig_prediction.add_trace(go.Scatter(x=extended_future_forecast_values.index, y=extended_future_forecast_values, mode='lines', name='Pronóstico Futuro', line=dict(color='green', dash='dot')))

# Línea vertical para marcar el fin del entrenamiento
date_for_vline = train_data.index[-1]

# Convertir la fecha a string para asegurar compatibilidad con plotly
x_vline_str = date_for_vline.strftime('%Y-%m-%d')

# fig_prediction.add_vline(
#     x=x_vline_str, 
#     line_width=1, 
#     line_dash="dot", 
#     line_color="red",
#     annotation_text="Fin Entrenamiento", 
#     annotation_position="top right",
#     annotation_font_color="red"
# )

fig_prediction.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Tasa de Excepción",
    title_text=f"Pronóstico de Tasa de Excepción para los Próximos {forecast_horizon} Días ({model_choice})",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_prediction, use_container_width=True)

st.subheader("Valores Pronosticados")
st.dataframe(future_forecast.to_frame(name='Tasa de Excepción Pronosticada'))