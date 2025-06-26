import streamlit as st
from utils.data_loader import cargar_avg_global
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import periodogram, find_peaks
from scipy.stats import boxcox
import numpy as np

st.set_page_config(
    page_title="Dashboard Analisis RPA",
    page_icon="🤖",
    layout="wide",
)

st.title("Supuestos de Datos RPA")

avg_global = cargar_avg_global()

st.header("Tasa de Excepción Global por Fecha")

fig_global = go.Figure(data=go.Scatter(x=avg_global.index, y=avg_global['tasa_excepcion'], mode='lines'))
fig_global.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Tasa de Excepción",
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

#####################################

st.header("Autocorrelación (ACF) de la Tasa de Excepción Global (Transformada con Box-Cox)")

serie_acf_original = avg_global['tasa_excepcion'].dropna()

# Para la transformación Box-Cox, los datos deben ser estrictamente positivos.
# Se suma una constante muy pequeña si hay ceros.
if (serie_acf_original <= 0).any():
    serie_positiva = serie_acf_original + 1e-6
else:
    serie_positiva = serie_acf_original

serie_acf, lmbda = boxcox(serie_positiva)

st.write(f"La transformación de Box-Cox se aplicó para estabilizar la varianza. Lambda (λ) = **{lmbda:.4f}**")

acf_values = acf(serie_acf, nlags=40, fft=True)

acf_values_no_lag0 = acf_values[1:]
lags_no_lag0 = range(1, len(acf_values_no_lag0) + 1)

acf_df = pd.DataFrame({'Lag': lags_no_lag0, 'ACF': acf_values_no_lag0})

conf_level = 1.96 / (len(serie_acf)**0.5)

fig_acf = go.Figure()

fig_acf.add_trace(go.Scatter(
    x=acf_df['Lag'],
    y=acf_df['ACF'],
    mode='markers',
    name='ACF',
    marker=dict(size=8, color='blue')
))

for i, (lag, acf_val) in acf_df.iterrows():
    fig_acf.add_trace(go.Scatter(
        x=[lag, lag],
        y=[0, acf_val],
        mode='lines',
        line=dict(color='blue', width=1),
        showlegend=False
    ))

fig_acf.add_trace(go.Scatter(
    x=[1, 40],
    y=[conf_level, conf_level],
    mode='lines',
    line=dict(color='red', dash='dash', width=1),
    name='Límite Superior 95% CI'
))
fig_acf.add_trace(go.Scatter(
    x=[1, 40],
    y=[-conf_level, -conf_level],
    mode='lines',
    line=dict(color='red', dash='dash', width=1),
    name='Límite Inferior 95% CI'
))

fig_acf.update_layout(
    xaxis_title="Lag",
    yaxis_title="Autocorrelación",
    title_text="Función de Autocorrelación (ACF)",
    showlegend=True,
    margin=dict(l=0, r=0, t=50, b=0),
    template="plotly_white"
)

st.plotly_chart(fig_acf, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

####################################

st.header("Análisis de Periodicidad (Periodograma)")

serie_periodograma = avg_global['tasa_excepcion'].dropna()
frequencies, power = periodogram(serie_periodograma, fs=1)

fig_periodograma = go.Figure()

fig_periodograma.add_trace(go.Scatter(
    x=frequencies, 
    y=power, 
    mode='lines', 
    name='Potencia Espectral',
    line=dict(color='dodgerblue')
))

height_threshold = np.mean(power)
peak_indices, _ = find_peaks(power, height=height_threshold)

labeled_periods = set()
for i in peak_indices:
    freq = frequencies[i]
    if freq == 0:
        continue

    period = 1 / freq
    rounded_period = round(period,1)
    
    if rounded_period not in labeled_periods and rounded_period > 1:
        pwr = power[i]
        fig_periodograma.add_trace(go.Scatter(
            x=[freq], 
            y=[pwr], 
            mode='markers', 
            marker=dict(color='purple', size=8, symbol='x'),
            showlegend=False
        ))
        fig_periodograma.add_annotation(
            x=freq,
            y=pwr,
            text=f"{rounded_period}d",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="purple")
        )
        labeled_periods.add(rounded_period)

fig_periodograma.add_vline(x=1/14, line_dash="dash", line_color="red", annotation_text="Período 14 días", annotation_position="top left")
fig_periodograma.add_vline(x=1/29, line_dash="dash", line_color="green", annotation_text="Período 29 días", annotation_position="top left")

fig_periodograma.update_layout(
    title_text='Periodograma de la Tasa de Excepción Global',
    xaxis_title='Frecuencia (Ciclos por día)',
    yaxis_title='Potencia',
    margin=dict(l=0, r=0, t=50, b=0),
    template="plotly_white",
    xaxis_range=[0, 0.20]
)

st.plotly_chart(fig_periodograma, use_container_width=True, config={'displayModeBar': False})

st.info("""
**Interpretación del Periodograma:**
- El periodograma ayuda a identificar las frecuencias dominantes en una serie temporal, lo que sugiere la presencia de ciclos.
- **Eje X (Frecuencia):** Representa ciclos por unidad de tiempo (en este caso, ciclos por día). Una frecuencia de 0.1 significa un ciclo cada 10 días (1/0.1).
- **Eje Y (Potencia):** Muestra la "fuerza" de cada frecuencia. Picos altos indican frecuencias importantes.
- **Picos Marcados:** Los picos en el gráfico corresponden a los períodos (1 / frecuencia) más significativos. Se han etiquetado los períodos redondeados a días enteros.
- **Conclusión:** Este análisis confirma visualmente los posibles ciclos en los datos. Los picos alrededor de las frecuencias correspondientes a 14 y 29 días refuerzan la idea de que existen patrones semanales y mensuales en la tasa de excepción.
""")

####################################

st.info("""
**Aclaración Importante: Estacionalidad vs. Estacionariedad**

Es crucial no confundir estos dos conceptos:
- **Estacionalidad:** Se refiere a patrones predecibles que se repiten en intervalos de tiempo fijos (ej. ciclos diarios, semanales, anuales). Una serie puede tener un componente estacional claro y aun así no ser estacionaria.
- **Estacionariedad:** Es una propiedad estadística de la serie. Una serie es estacionaria si su media, varianza y autocorrelación son constantes a lo largo del tiempo. La presencia de una tendencia o de estacionalidad fuerte a menudo implica que la serie no es estacionaria.

El siguiente test (Dickey-Fuller Aumentado) evalúa la **estacionariedad**, no la estacionalidad.
""")

# --- Sección 3: Test de Estacionariedad (ADF) ---
st.header("Test de Estacionariedad (Dickey-Fuller Aumentado)")

serie_adf = avg_global['tasa_excepcion'].dropna()

max_lags_adf = 50

# 1. Realizar el test con selección automática de lags (AIC) para un resultado rápido
st.subheader("Resultado del Test ADF (Lag Óptimo según AIC)")
best_adf_result = adfuller(serie_adf, maxlag=max_lags_adf, autolag='AIC')

col1, col2, col3 = st.columns(3)
col1.metric("Estadístico ADF", f"{best_adf_result[0]:.4f}")
col2.metric("P-valor", f"{best_adf_result[1]:.4f}")
col3.metric("Lags Usados", best_adf_result[2])

if best_adf_result[1] < 0.05:
    st.success(f"**Conclusión:** El p-valor ({best_adf_result[1]:.4f}) es menor que 0.05. Se rechaza la hipótesis nula (H0). **La serie es probablemente estacionaria.** Esto significa que sus propiedades estadísticas (como la media y la varianza) no cambian con el tiempo.")
else:
    st.warning(f"**Conclusión:** El p-valor ({best_adf_result[1]:.4f}) es mayor o igual a 0.05. No se puede rechazar la hipótesis nula (H0). **La serie es probablemente no estacionaria.** Esto sugiere la presencia de una **tendencia**, una estacionalidad fuerte u otra estructura que varía con el tiempo, lo cual es consistente con lo observado en el gráfico de la serie.")

# 2. Generar el gráfico de p-valores para cada lag
st.subheader(f"Análisis de P-valores del Test ADF para Diferentes Lags (1 a {max_lags_adf})")

p_values = []
lags_range = range(1, max_lags_adf + 1)

for i in lags_range:
    try:
        # Forzamos el test para cada lag específico usando autolag=None
        result = adfuller(serie_adf, maxlag=i, autolag=None)
        p_values.append(result[1])
    except Exception as e:
        p_values.append(None) # En caso de algún error, añadimos None

adf_pvals_df = pd.DataFrame({'Lag': list(lags_range), 'P-Value': p_values})

# Crear el gráfico de p-valores, similar al de ACF
fig_adf_pvals = go.Figure()

# Añadir markers (puntos)
fig_adf_pvals.add_trace(go.Scatter(
    x=adf_pvals_df['Lag'],
    y=adf_pvals_df['P-Value'],
    mode='markers',
    name='P-valor',
    marker=dict(size=8, color='darkorange')
))

# Añadir stems (líneas verticales)
for i, row in adf_pvals_df.iterrows():
    if pd.notna(row['P-Value']):
        fig_adf_pvals.add_trace(go.Scatter(
            x=[row['Lag'], row['Lag']],
            y=[0, row['P-Value']],
            mode='lines',
            line=dict(color='darkorange', width=1),
            showlegend=False
        ))

# Añadir línea de significancia de 0.05
fig_adf_pvals.add_hline(
    y=0.05, 
    line_dash="dash", 
    line_color="red",
    annotation_text="Significancia (α=0.05)", 
    annotation_position="bottom right"
)

fig_adf_pvals.update_layout(
    xaxis_title="Número de Lags",
    yaxis_title="P-valor",
    title_text="P-valores del Test ADF por Número de Lags",
    showlegend=True,
    margin=dict(l=0, r=0, t=50, b=0),
    template="plotly_white",
    yaxis=dict(range=[0, max(1.0, adf_pvals_df['P-Value'].max() * 1.1 if not adf_pvals_df.empty else 1.0)]) # Ajustar eje Y
)

st.plotly_chart(fig_adf_pvals, use_container_width=True, config={'displayModeBar': False})

st.info("""
**Interpretación del Gráfico:**
- **Hipótesis Nula (H0):** La serie temporal tiene una raíz unitaria (es no estacionaria).
- **Eje Y:** Muestra el p-valor del test para un número de lags específico (Eje X).
- **Línea Roja (Nivel de Significancia α=0.05):** Es el umbral de decisión.
- **Conclusión:** Si los puntos naranjas (p-valores) están **por debajo** de la línea roja, se rechaza la hipótesis nula, sugiriendo que la serie es estacionaria para ese número de lags. Si están **por encima**, no hay evidencia suficiente para rechazar la no estacionariedad.
""")



################################

st.header("Descomposición Estacional de la Tasa de Excepción Global")

serie_a_descomponer = avg_global['tasa_excepcion'].dropna()

periodo_estacional = st.slider(
    'Días para la descomposición estacional',
    min_value=10,
    max_value=30,
    value=14,
    step=1,
    help="Selecciona el período (en días) para realizar la descomposición estacional de la serie temporal."
)

decomposition = seasonal_decompose(
    serie_a_descomponer,
    model='additive',
    period=periodo_estacional
)

st.subheader("Serie Original (Observada)")
fig_observed = go.Figure(data=go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines'))
fig_observed.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_observed, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

st.subheader("Tendencia")
fig_trend = go.Figure(data=go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines'))
fig_trend.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

st.subheader("Componente Estacional")
fig_seasonal = go.Figure(data=go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines'))
fig_seasonal.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_seasonal, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

st.subheader("Residuos")
fig_resid = go.Figure(data=go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines'))
fig_resid.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_resid, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})