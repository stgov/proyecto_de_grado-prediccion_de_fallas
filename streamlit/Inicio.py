import streamlit as st

st.set_page_config(
    page_title="Dashboard Analisis RPA",
    page_icon="🤖",
    layout="wide",
)

st.title("Dashboard de Análisis y Predicción de Fallas RPA")

# Información del proyecto
st.markdown("""
## 🤖 Bienvenido al Sistema de Predicción de Fallas RPA

Esta aplicación está diseñada para analizar y predecir fallas en procesos de Automatización Robótica de Procesos (RPA) 
utilizando técnicas avanzadas de análisis de series de tiempo y modelamiento estadístico.

**Proyecto desarrollado para el área de Robotics del Banco Itaú**

### 👥 Equipo de Desarrollo
- **Bruno Cerda**
- **Santiago Valenzuela**

*Agradecemos especialmente la colaboración de Eduardo Cancino y Jonathan García*

### 📊 Funcionalidades de la Aplicación

La aplicación consta de **4 módulos principales** que te permiten realizar un análisis completo desde la exploración 
de datos hasta la predicción de fallas futuras:
""")

# Sección de páginas
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🔍 **Análisis Exploratorio**
    - Visualización interactiva de datos RPA
    - Análisis de tendencias y patrones temporales
    - Estadísticas descriptivas de las tasas de excepción
    - Comparación entre diferentes robots y procesos
    - Identificación de anomalías y comportamientos atípicos
    
    #### ✅ **Supuestos de Datos**
    - Verificación de estacionariedad de las series
    - Análisis de descomposición temporal (tendencia, estacionalidad)
    - Transformaciones de estabilización (Box-Cox)
    - Tests estadísticos de validación
    - Análisis de autocorrelación y periodicidad
    """)

with col2:
    st.markdown("""
    #### 🔮 **Predicción**
    - Pronósticos futuros de tasas de excepción
    - Intervalos de confianza para las predicciones
    - Visualización interactiva de resultados
    - Configuración de horizontes de predicción personalizados
    
    #### 📈 **Modelamiento**
    - Implementación de modelos ETS (Error, Trend, Seasonality)
    - Entrenamiento con validación temporal
    - Evaluación de métricas de rendimiento (MAE, MSE, R²)
    - Análisis de residuos para validación del modelo
    - Diagnóstico de la calidad del ajuste
    """)

# Información adicional
st.markdown("""
---
### 🎯 Objetivo del Proyecto

El objetivo principal es desarrollar un sistema predictivo que permita:
- **Anticipar fallas** en procesos RPA antes de que ocurran
- **Optimizar la disponibilidad** de los robots automatizados
- **Reducir tiempos de inactividad** mediante predicciones tempranas
- **Mejorar la eficiencia operacional** de los procesos automatizados

### 📝 Datos Utilizados

La aplicación analiza datos de logs de trabajo de robots RPA que incluyen:
- Tasas de excepción por robot y proceso
- Métricas globales agregadas temporalmente
- Información de rendimiento histórico
- Patrones de comportamiento temporal

### 🚀 Comenzar el Análisis

Para empezar, navega a través de las páginas en el orden sugerido:
1. **Análisis Exploratorio** - Para familiarizarte con los datos
2. **Supuestos de Datos** - Para validar las condiciones de modelamiento
3. **Modelamiento** - Para entrenar y evaluar el modelo predictivo
4. **Predicción** - Para generar pronósticos futuros

¡Utiliza la barra lateral para navegar entre las diferentes secciones!
""")

# Footer
st.markdown("---")
st.markdown("*Desarrollado para el análisis y predicción de fallas en sistemas RPA*")

