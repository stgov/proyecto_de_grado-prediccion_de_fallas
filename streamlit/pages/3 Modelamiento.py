import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

from utils.data_loader import cargar_avg_global
from utils.modeling_utils import train_ets_model

st.set_page_config(
    page_title="Dashboard Analisis RPA",
    page_icon="🤖",
    layout="wide",
)

st.title("Modelamiento de Series de Tiempo (ETS)")
st.write("En esta sección, se aplicará un modelo de Suavizado Exponencial (ETS) a la tasa de excepción global para realizar pronósticos.")

data = cargar_avg_global()
serie_a_modelar = data['tasa_excepcion'].dropna()

st.subheader("Configuración del Modelo y Pronóstico")

train_percentage = st.slider("Porcentaje de Datos para Entrenamiento:", min_value=50, max_value=90, value=50, step=1)

train_size = int(len(serie_a_modelar) * (train_percentage / 100))
train_data = serie_a_modelar.iloc[:train_size]
test_data = serie_a_modelar.iloc[train_size:]

st.write(f"**Serie de tiempo total:** {len(serie_a_modelar)} puntos")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Datos de entrenamiento:** {len(train_data)} puntos")
    st.write(f"**Período:** {train_data.index.min().strftime('%Y-%m-%d')} a {train_data.index.max().strftime('%Y-%m-%d')}")

with col2:
    st.write(f"**Datos de prueba:** {len(test_data)} puntos")
    st.write(f"**Período:** {test_data.index.min().strftime('%Y-%m-%d')} a {test_data.index.max().strftime('%Y-%m-%d')}")

st.markdown("##### Entrenamiento del Modelo Holt-Winters Exponential Smoothing")

seasonal_periods = st.slider(
    "Período Estacional (número de puntos por ciclo):",
    min_value=1,
    max_value=30,
    value=14,
    step=1,
)

model = train_ets_model(train_data, seasonal_periods)
st.success("Modelo ETS entrenado exitosamente.")

# Store model and data in session state for the prediction page
st.session_state['ets_model'] = model
st.session_state['serie_a_modelar'] = serie_a_modelar
st.session_state['train_data'] = train_data
st.session_state['test_data'] = test_data

forecast_steps = len(test_data)
forecast = model.predict(start=test_data.index.min(), end=test_data.index.max())

st.subheader("Pronóstico vs Datos Reales")

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Datos de Entrenamiento', line=dict(color='blue')))

last_train_index = train_data.index[-1]
last_train_value = train_data.iloc[-1]

extended_test_values = pd.concat([pd.Series([last_train_value], index=[last_train_index]), test_data])
extended_test_values = extended_test_values.loc[~extended_test_values.index.duplicated(keep='last')]

fig_forecast.add_trace(go.Scatter(x=extended_test_values.index, y=extended_test_values, mode='lines', name='Datos Reales (Prueba)', line=dict(color='orange')))

last_fitted_value = model.fittedvalues.iloc[-1]
extended_forecast_index = pd.Index([train_data.index[-1]]).union(forecast.index)
extended_forecast_values = pd.concat([pd.Series([last_fitted_value], index=[train_data.index[-1]]), forecast])
extended_forecast_values = extended_forecast_values.loc[~extended_forecast_values.index.duplicated(keep='last')]

fig_forecast.add_trace(go.Scatter(x=extended_forecast_values.index, y=extended_forecast_values, mode='lines', name='Pronóstico ETS', line=dict(color='green', dash='dot')))

fig_forecast.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Tasa de Excepción",
    title_text="Pronóstico del Modelo ETS",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Evaluación del Modelo")

mae_test = mean_absolute_error(test_data, forecast)
mse_test = mean_squared_error(test_data, forecast)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(test_data, forecast)

actual_change = test_data.diff().dropna()
predicted_change = forecast.diff().dropna()

common_index = actual_change.index.intersection(predicted_change.index)
actual_change_aligned = actual_change.loc[common_index]
predicted_change_aligned = predicted_change.loc[common_index]

if len(common_index) > 0:
    correct_direction_predictions = np.sum(np.sign(actual_change_aligned) == np.sign(predicted_change_aligned))
    directional_accuracy = (correct_direction_predictions / len(common_index)) * 100
else:
    directional_accuracy = 0.0

train_forecast = model.fittedvalues
mae_train = mean_absolute_error(train_data, train_forecast)
mse_train = mean_squared_error(train_data, train_forecast)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(train_data, train_forecast)

st.markdown("##### Métricas de Rendimiento")

col_names_1, col_names_2 = st.columns(2)
with col_names_1:
    st.markdown("###### Conjunto de Entrenamiento")
with col_names_2:
    st.markdown("###### Conjunto de Prueba")

metric_cols = st.columns(2)
with metric_cols[0]:
    st.metric("MAE (Train)", f"{mae_train:.4f}")
    st.metric("MSE (Train)", f"{mse_train:.4f}")
    st.metric("RMSE (Train)", f"{rmse_train:.4f}")
    st.metric("R2 Score (Train)", f"{r2_train:.4f}")
with metric_cols[1]:
    st.metric("MAE (Test)", f"{mae_test:.4f}")
    st.metric("MSE (Test)", f"{mse_test:.4f}")
    st.metric("RMSE (Test)", f"{rmse_test:.4f}")
    st.metric("R2 Score (Test)", f"{r2_test:.4f}")

st.markdown("##### Precisión Direccional (Solo en Conjunto de Prueba)")
st.metric("Precisión Direccional (Test)", f"{directional_accuracy:.2f}%")

st.subheader("Análisis de Residuos")
st.write("Un buen modelo de series de tiempo debe tener residuos que sean aleatorios, no correlacionados, con media cero y varianza constante (homocedasticidad), y normalmente distribuidos. A continuación se analizan los residuos del modelo sobre el **conjunto de entrenamiento** para diagnosticar la calidad del ajuste.")

residuals = model.resid

fig_residuals_time = go.Figure(data=go.Scatter(x=residuals.index, y=residuals, mode='lines'))
fig_residuals_time.add_hline(y=0, line_dash="dot", line_color="red")
fig_residuals_time.update_layout(
    title_text='Residuos vs Tiempo',
    xaxis_title='Fecha',
    yaxis_title='Residuos',
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_residuals_time, use_container_width=True)

fig_hist_residuals = px.histogram(residuals, nbins=15, title='Distribución de Residuos')
fig_hist_residuals.update_layout(
    xaxis_title='Residuos',
    yaxis_title='Frecuencia',
    bargap=0.05,
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_hist_residuals, use_container_width=True)

sorted_residuals = np.sort(residuals.values)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))

fig_qq = go.Figure(data=go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers'))
fig_qq.add_trace(go.Scatter(x=[min(theoretical_quantiles), max(theoretical_quantiles)],
                             y=[min(theoretical_quantiles), max(theoretical_quantiles)],
                             mode='lines', line=dict(color='red', dash='dash'), name='Línea Teórica'))
fig_qq.update_layout(
    title_text='Q-Q Plot de Residuos',
    xaxis_title='Cuantiles Teóricos Normales',
    yaxis_title='Cuantiles de Residuos Ordenados',
    showlegend=False,
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_qq, use_container_width=True)

fig_resid_vs_fitted = go.Figure(data=go.Scatter(x=model.fittedvalues, y=residuals, mode='markers'))
fig_resid_vs_fitted.add_hline(y=0, line_dash="dot", line_color="red")
fig_resid_vs_fitted.update_layout(
    title_text='Residuos vs Valores Ajustados',
    xaxis_title='Valores Ajustados',
    yaxis_title='Residuos',
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_resid_vs_fitted, use_container_width=True)

st.subheader("ACF de los Residuos")
st.write("La ACF de los residuos debería mostrar poca o ninguna autocorrelación significativa, indicando que el modelo ha capturado la mayor parte de la información de la serie.")

serie_acf_resid = residuals.dropna()
acf_values_resid = acf(serie_acf_resid, nlags=40, fft=True)
acf_values_resid_no_lag0 = acf_values_resid[1:]
lags_resid_no_lag0 = range(1, len(acf_values_resid_no_lag0) + 1)

acf_df_resid = pd.DataFrame({'Lag': lags_resid_no_lag0, 'ACF': acf_values_resid_no_lag0})
conf_level_resid = 1.96 / (serie_acf_resid.shape[0]**0.5)

fig_acf_resid = go.Figure()
fig_acf_resid.add_trace(go.Scatter(
    x=acf_df_resid['Lag'],
    y=acf_df_resid['ACF'],
    mode='markers',
    name='ACF Residuos',
    marker=dict(size=8, color='blue')
))

for i, (lag, acf_val) in acf_df_resid.iterrows():
    fig_acf_resid.add_trace(go.Scatter(
        x=[lag, lag],
        y=[0, acf_val],
        mode='lines',
        line=dict(color='blue', width=1),
        showlegend=False
    ))

fig_acf_resid.add_trace(go.Scatter(
    x=[1, 40],
    y=[conf_level_resid, conf_level_resid],
    mode='lines',
    line=dict(color='red', dash='dash', width=1),
    name='Límite Superior 95% CI'
))
fig_acf_resid.add_trace(go.Scatter(
    x=[1, 40],
    y=[-conf_level_resid, -conf_level_resid],
    mode='lines',
    line=dict(color='red', dash='dash', width=1),
    name='Límite Inferior 95% CI'
))

fig_acf_resid.update_layout(
    xaxis_title="Lag",
    yaxis_title="Autocorrelación",
    title_text="Función de Autocorrelación (ACF) de Residuos",
    showlegend=True,
    margin=dict(l=0, r=0, t=50, b=0),
    template="plotly_white"
)
st.plotly_chart(fig_acf_resid, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})