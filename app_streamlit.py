"""
Aplicación Streamlit para visualización de Predicción de Temperaturas con LSTM
Proyecto: Predicción de Temperaturas Mínimas - Melbourne, Australia (1981-1990)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import os
from pathlib import Path

# Configuración de la página
st.set_page_config( page_title="Predicción de Temperaturas con LSTM", page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# Funciones auxiliares y cache

@st.cache_data
def cargaPredicciones():
    """Carga el archivo de predicciones"""
    try:
        df = pd.read_csv('outputs/predictions.csv')
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo predictions.csv")
        return None

@st.cache_data
def load_metrics():
    """Carga las métricas del modelo"""
    try:
        df = pd.read_csv('outputs/final_metrics.csv')
        return df.iloc[0].to_dict()
    except FileNotFoundError:
        st.error("No se encontró el archivo final_metrics.csv")
        return None

@st.cache_data
def load_dataset():
    """Carga el dataset original"""
    try:
        df = pd.read_csv('data/1_Daily_minimum_temps.xls')
        df['Temp'] = df['Temp'].astype(str).str.replace('?', '-', regex=False)
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        df = df.dropna()
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar dataset: {e}")
        return None

@st.cache_resource
def load_model_and_scaler():
    """Carga el modelo entrenado y el scaler"""
    try:
        # Intentar importar TensorFlow
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            st.error("TensorFlow no está instalado. Por favor, ejecuta: `pip install tensorflow`")
            st.info("Luego reinicia la aplicación Streamlit presionando `Ctrl+C` y ejecutando de nuevo `python -m streamlit run app_streamlit.py`")
            return None, None
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Cargar modelo
        model = load_model('outputs/best_model_final.keras')
        
        # Recrear scaler con los mismos datos de entrenamiento
        df = load_dataset()
        if df is not None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[['Temp']].values)
            return model, scaler
        return None, None
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo del modelo: 'outputs/best_model_final.keras'")
        st.info("💡 Asegúrate de que el modelo está en la carpeta 'outputs/'")
        return None, None
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        st.info("💡 Si el problema persiste, intenta reinstalar TensorFlow: `pip install tensorflow --force-reinstall`")
        return None, None

def make_prediction(sequence, model, scaler):
    """
    Realiza una predicción dado una secuencia de 60 valores
    
    Args:
        sequence: Array de 60 valores de temperatura
        model: Modelo LSTM cargado
        scaler: MinMaxScaler para normalización
    
    Returns:
        Predicción desnormalizada
    """
    # Normalizar la secuencia
    sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
    
    # Reshape para LSTM [samples, time_steps, features]
    sequence_reshaped = sequence_scaled.reshape(1, 60, 1)
    
    # Hacer predicción
    prediction_scaled = model.predict(sequence_reshaped, verbose=0)
    
    # Desnormalizar
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return prediction[0][0]

# ============================================================================
# HEADER Y TÍTULO
# ============================================================================

st.markdown('<h1 class="main-header">🌡️ Predicción de Temperaturas con LSTM</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #e8eef7; border-radius: 10px; margin-bottom: 2rem; border: 1px solid #667eea;'>
    <h3 style='color: #333;'>📊 Análisis de Series Temporales</h3>
    <p style='font-size: 1.1rem; color: #444;'>
        <b>Dataset:</b> Temperaturas Mínimas Diarias - Melbourne, Australia (1981-1990)<br>
        <b>Observaciones:</b> 3,650 días | <b>Modelo:</b> LSTM Optimizado
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎛️ Panel de Control")
st.sidebar.markdown("---")

# Selector de sección
section = st.sidebar.radio(
    "Navegar a:",
    ["📊 Métricas y Resultados", "🎯 Predicción Interactiva", "📈 Visualizaciones", "⚙️ Configuración del Modelo"],
    label_visibility="visible"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Sobre este proyecto:**

Este modelo LSTM fue entrenado para predecir temperaturas mínimas diarias usando una ventana de 60 días consecutivos.

**Características:**
- Sequence Length: 60 días
- LSTM Units: [64]
- Dropout: 0.3
- Learning Rate: 0.001
""")

# ============================================================================
# SECCIÓN 1: MÉTRICAS Y RESULTADOS
# ============================================================================

if section == "📊 Métricas y Resultados":
    
    st.header("📊 Métricas de Evaluación del Modelo")
    
    # Cargar métricas
    metrics = load_metrics()
    
    if metrics:
        # Métricas principales en columnas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📉 RMSE",
                value=f"{metrics['RMSE']:.2f}°C",
                delta=None,
                help="Root Mean Squared Error - Menor es mejor"
            )
        
        with col2:
            st.metric(
                label="📊 MAE",
                value=f"{metrics['MAE']:.2f}°C",
                delta=None,
                help="Mean Absolute Error - Menor es mejor"
            )
        
        with col3:
            st.metric(
                label="🎯 R²",
                value=f"{metrics['R2']:.4f}",
                delta=None,
                help="Coeficiente de Determinación - Cercano a 1 es mejor"
            )
        
        with col4:
            st.metric(
                label="📈 MAPE",
                value=f"{metrics['MAPE']:.2f}%",
                delta=None,
                help="Mean Absolute Percentage Error - Menor es mejor"
            )
        
        st.markdown("---")
        
        # Interpretación de métricas
        st.subheader("🔍 Interpretación de Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics['MAPE'] < 10:
                st.success("✅ **MAPE < 10%:** Predicciones muy precisas")
            elif metrics['MAPE'] < 20:
                st.info("ℹ️ **MAPE 10-20%:** Predicciones buenas")
            else:
                st.warning("⚠️ **MAPE > 20%:** Predicciones con margen de mejora")
        
        with col2:
            if metrics['R2'] > 0.9:
                st.success("✅ **R² > 0.9:** Excelente ajuste del modelo")
            elif metrics['R2'] > 0.7:
                st.info("ℹ️ **R² 0.7-0.9:** Buen ajuste del modelo")
            else:
                st.warning("⚠️ **R² < 0.7:** Ajuste del modelo mejorable")
        
        st.markdown("---")
        
        # Gráficos de predicciones
        st.subheader("📈 Predicciones vs Valores Reales")
        
        predictions_df = cargaPredicciones()
        
        if predictions_df is not None:
            # Gráfico interactivo con Plotly
            fig = go.Figure()
            
            # Valores reales
            fig.add_trace(go.Scatter(
                y=predictions_df['real'],
                mode='lines',
                name='Valores Reales',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='Real: %{y:.2f}°C<extra></extra>'
            ))
            
            # Predicciones
            fig.add_trace(go.Scatter(
                y=predictions_df['prediccion'],
                mode='lines',
                name='Predicciones',
                line=dict(color='#E94F37', width=2, dash='dash'),
                hovertemplate='Predicción: %{y:.2f}°C<extra></extra>'
            ))
            
            # Área de error
            fig.add_trace(go.Scatter(
                y=predictions_df['real'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                y=predictions_df['prediccion'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                fillcolor='rgba(128,128,128,0.2)',
                name='Área de Error',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title='Serie Temporal - Predicciones del Modelo LSTM',
                xaxis_title='Índice de Muestra',
                yaxis_title='Temperatura (°C)',
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de dispersión
            st.subheader("🎯 Gráfico de Dispersión")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Scatter plot
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=predictions_df['real'],
                    y=predictions_df['prediccion'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=predictions_df['error'].abs(),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Error Abs."),
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate='Real: %{x:.2f}°C<br>Pred: %{y:.2f}°C<extra></extra>'
                ))
                
                # Línea de predicción perfecta
                min_val = min(predictions_df['real'].min(), predictions_df['prediccion'].min())
                max_val = max(predictions_df['real'].max(), predictions_df['prediccion'].max())
                
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Predicción Perfecta',
                    hoverinfo='skip'
                ))
                
                fig_scatter.update_layout(
                    title=f'Predicciones vs Valores Reales (R² = {metrics["R2"]:.4f})',
                    xaxis_title='Valores Reales (°C)',
                    yaxis_title='Predicciones (°C)',
                    height=500,
                    template='plotly_white'
                )
                
                fig_scatter.update_xaxes(scaleanchor="y", scaleratio=1)
                fig_scatter.update_yaxes(scaleanchor="x", scaleratio=1)
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Estadísticas de Error")
                st.metric("Error Máximo", f"{predictions_df['error'].abs().max():.2f}°C")
                st.metric("Error Medio", f"{predictions_df['error'].mean():.2f}°C")
                st.metric("Desv. Est. Error", f"{predictions_df['error'].std():.2f}°C")
                
                # Distribución de errores
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=predictions_df['error'],
                    nbinsx=30,
                    marker_color='#667eea',
                    opacity=0.7
                ))
                fig_hist.update_layout(
                    title='Distribución de Errores',
                    xaxis_title='Error (°C)',
                    yaxis_title='Frecuencia',
                    height=300,
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# SECCIÓN 2: PREDICCIÓN INTERACTIVA
# ============================================================================

elif section == "🎯 Predicción Interactiva":
    
    st.header("🎯 Haz tu Propia Predicción")
    
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='margin: 0; color: black;'>
            ℹ️ <b>Nota:</b> El modelo LSTM necesita una secuencia de <b>60 días consecutivos</b> 
            de temperaturas mínimas para predecir el siguiente día.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo y scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("❌ No se pudo cargar el modelo. Verifica que existe el archivo 'outputs/best_model_final.keras'")
    else:
        # Tabs para los dos modos de predicción
        st.markdown("""
        <style>
            .stTabs [data-baseweb="tab"] p {
                color: black !important;
            }
        </style>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🚀 Predicción Rápida", "✏️ Predicción Personalizada"])
        
        # ========== TAB 1: PREDICCIÓN RÁPIDA ==========
        with tab1:
            st.subheader("🚀 Predicción Rápida desde el Dataset")
            
            df_original = load_dataset()
            
            if df_original is not None:
                st.markdown("Selecciona una fecha del dataset para predecir el siguiente día:")
                
                # Fechas disponibles (después del día 60 para tener secuencia)
                available_dates = df_original.index[60:]
                
                selected_date = st.date_input(
                    "Fecha de referencia:",
                    value=available_dates[100],
                    min_value=available_dates[0].date(),
                    max_value=available_dates[-1].date(),
                    help="Se usarán los 60 días anteriores a esta fecha para hacer la predicción"
                )
                
                if st.button("🔮 Predecir", key="predict_quick", type="primary"):
                    # Convertir a datetime
                    selected_datetime = pd.to_datetime(selected_date)
                    
                    # Obtener el índice en el dataset
                    try:
                        idx = df_original.index.get_loc(selected_datetime)
                        
                        if idx < 60:
                            st.error("⚠️ No hay suficientes datos históricos para esta fecha. Selecciona una fecha posterior.")
                        else:
                            # Obtener secuencia de 60 días
                            sequence = df_original['Temp'].iloc[idx-60:idx].values
                            
                            # Hacer predicción
                            with st.spinner('Calculando predicción...'):
                                prediction = make_prediction(sequence, model, scaler)
                            
                            # Obtener valor real si existe
                            real_value = None
                            if idx < len(df_original):
                                real_value = df_original['Temp'].iloc[idx]
                            
                            # Mostrar resultados
                            st.success("✅ Predicción completada!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="🔮 Predicción",
                                    value=f"{prediction:.2f}°C",
                                    help="Temperatura mínima predicha para el día siguiente"
                                )
                            
                            if real_value is not None:
                                error = real_value - prediction
                                with col2:
                                    st.metric(
                                        label="📊 Valor Real",
                                        value=f"{real_value:.2f}°C",
                                        delta=None
                                    )
                                with col3:
                                    st.metric(
                                        label="📉 Error",
                                        value=f"{abs(error):.2f}°C",
                                        delta=f"{error:.2f}°C",
                                        delta_color="inverse"
                                    )
                            
                            # Visualización de la secuencia
                            st.markdown("---")
                            st.subheader("📊 Secuencia de Entrada (60 días)")
                            
                            # Crear fechas para la secuencia
                            seq_dates = df_original.index[idx-60:idx]
                            
                            fig_seq = go.Figure()
                            
                            fig_seq.add_trace(go.Scatter(
                                x=seq_dates,
                                y=sequence,
                                mode='lines+markers',
                                name='Temperaturas (60 días)',
                                line=dict(color='#2E86AB', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Agregar predicción
                            next_date = selected_datetime + pd.Timedelta(days=1)
                            fig_seq.add_trace(go.Scatter(
                                x=[seq_dates[-1], next_date],
                                y=[sequence[-1], prediction],
                                mode='lines+markers',
                                name='Predicción',
                                line=dict(color='#E94F37', width=3, dash='dash'),
                                marker=dict(size=10, symbol='star')
                            ))
                            
                            # Agregar valor real si existe
                            if real_value is not None:
                                fig_seq.add_trace(go.Scatter(
                                    x=[next_date],
                                    y=[real_value],
                                    mode='markers',
                                    name='Valor Real',
                                    marker=dict(size=10, color='green', symbol='diamond')
                                ))
                            
                            fig_seq.update_layout(
                                title='Secuencia de 60 días + Predicción',
                                xaxis_title='Fecha',
                                yaxis_title='Temperatura (°C)',
                                height=400,
                                hovermode='x unified',
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_seq, use_container_width=True)
                    
                    except KeyError:
                        st.error("❌ La fecha seleccionada no existe en el dataset.")
        
        # ========== TAB 2: PREDICCIÓN PERSONALIZADA ==========
        with tab2:
            st.subheader("✏️ Predicción con Datos Personalizados")
            
            st.markdown("""
            Ingresa 60 valores de temperatura mínima diaria (en °C) separados por comas.
            
            **Ejemplo:**
            ```
            10.5, 11.2, 9.8, 12.1, 13.4, 14.2, ...
            ```
            """)
            
            # Campo de texto para ingresar valores
            user_input = st.text_area(
                "Ingresa 60 valores de temperatura (separados por comas):",
                height=150,
                placeholder="10.5, 11.2, 9.8, 12.1, 13.4, ...",
                help="Deben ser exactamente 60 valores numéricos separados por comas"
            )
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                predict_button = st.button("🔮 Predecir", key="predict_custom", type="primary")
            
            if predict_button:
                if not user_input.strip():
                    st.error("❌ Por favor ingresa los valores de temperatura.")
                else:
                    try:
                        # Parsear valores
                        values = [float(x.strip()) for x in user_input.split(',')]
                        
                        # Validar cantidad
                        if len(values) != 60:
                            st.error(f"❌ Se necesitan exactamente 60 valores. Ingresaste {len(values)}.")
                        else:
                            # Validar rango razonable
                            if any(v < -50 or v > 50 for v in values):
                                st.warning("⚠️ Algunos valores parecen estar fuera del rango típico de temperaturas. ¿Estás seguro?")
                            
                            # Convertir a numpy array
                            sequence = np.array(values)
                            
                            # Hacer predicción
                            with st.spinner('Calculando predicción...'):
                                prediction = make_prediction(sequence, model, scaler)
                            
                            # Mostrar resultado
                            st.success("✅ Predicción completada!")
                            
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                                <h2 style='margin: 0;'>🔮 Predicción del Día 61</h2>
                                <h1 style='font-size: 3.5rem; margin: 1rem 0;'>{prediction:.2f}°C</h1>
                                <p style='margin: 0; opacity: 0.9;'>Temperatura mínima estimada</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Visualización de la secuencia ingresada
                            st.subheader("📊 Visualización de tu Secuencia")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig_custom = go.Figure()
                                
                                fig_custom.add_trace(go.Scatter(
                                    y=sequence,
                                    mode='lines+markers',
                                    name='Temperaturas Ingresadas',
                                    line=dict(color='#2E86AB', width=2),
                                    marker=dict(size=4),
                                    hovertemplate='Día %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
                                ))
                                
                                # Agregar predicción
                                fig_custom.add_trace(go.Scatter(
                                    x=[59, 60],
                                    y=[sequence[-1], prediction],
                                    mode='lines+markers',
                                    name='Predicción',
                                    line=dict(color='#E94F37', width=3, dash='dash'),
                                    marker=dict(size=10, symbol='star'),
                                    hovertemplate='Día %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
                                ))
                                
                                fig_custom.update_layout(
                                    title='Secuencia de 60 días + Predicción',
                                    xaxis_title='Día',
                                    yaxis_title='Temperatura (°C)',
                                    height=400,
                                    hovermode='closest',
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig_custom, use_container_width=True)
                            
                            with col2:
                                st.markdown("### 📈 Estadísticas")
                                st.metric("Temperatura Promedio", f"{np.mean(sequence):.2f}°C")
                                st.metric("Temperatura Máxima", f"{np.max(sequence):.2f}°C")
                                st.metric("Temperatura Mínima", f"{np.min(sequence):.2f}°C")
                                st.metric("Desv. Estándar", f"{np.std(sequence):.2f}°C")
                                
                                # Tendencia
                                trend = "↗️ Ascendente" if sequence[-1] > sequence[0] else "↘️ Descendente"
                                st.metric("Tendencia", trend)
                    
                    except ValueError:
                        st.error("❌ Error al procesar los valores. Asegúrate de ingresar solo números separados por comas.")

# ============================================================================
# SECCIÓN 3: VISUALIZACIONES
# ============================================================================

elif section == "📈 Visualizaciones":
    
    st.header("📈 Visualizaciones del Modelo")
    
    # Verificar existencia de imágenes
    images_dir = Path('outputs')
    
    available_images = {
        'Predicciones Optimizadas': 'predictions_optimized.png',
        'Gráfico de Dispersión': 'scatter_optimized.png',
        'Análisis de Errores': 'error_analysis_optimized.png',
        'Historial de Entrenamiento': 'training_history_optimized.png',
        'Comparación de Optimización': 'optimization_comparison.png'
    }
    
    # Crear tabs para las imágenes
    tabs = st.tabs(list(available_images.keys()))
    
    for tab, (title, filename) in zip(tabs, available_images.items()):
        with tab:
            image_path = images_dir / filename
            if image_path.exists():
                image = Image.open(image_path)
                st.image(image, use_container_width=True, caption=title)
            else:
                st.warning(f"⚠️ No se encontró la imagen: {filename}")
    
    # Tabla de resultados de optimización
    st.markdown("---")
    st.subheader("📊 Tabla de Resultados de Optimización")
    
    try:
        opt_results = pd.read_csv('outputs/optimization_results.csv')
        st.dataframe(opt_results, use_container_width=True, height=400)
        
        # Gráfico interactivo de resultados
        st.subheader("📈 Resultados de Grid Search")
        
        fig_opt = go.Figure()
        
        fig_opt.add_trace(go.Scatter(
            x=list(range(len(opt_results))),
            y=opt_results['RMSE'],
            mode='lines+markers',
            name='RMSE',
            marker=dict(size=8, color=opt_results['RMSE'], colorscale='Viridis', showscale=True)
        ))
        
        fig_opt.update_layout(
            title='RMSE por Configuración de Hiperparámetros',
            xaxis_title='Configuración',
            yaxis_title='RMSE',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_opt, use_container_width=True)
        
    except FileNotFoundError:
        st.info("ℹ️ No se encontró el archivo de resultados de optimización.")

# ============================================================================
# SECCIÓN 4: CONFIGURACIÓN DEL MODELO
# ============================================================================

elif section == "⚙️ Configuración del Modelo":
    
    st.header("⚙️ Configuración del Modelo LSTM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Hiperparámetros Óptimos")
        
        config_data = {
            "Parámetro": [
                "Sequence Length",
                "LSTM Units",
                "Dropout Rate",
                "Learning Rate",
                "Dense Units",
                "Epochs",
                "Batch Size"
            ],
            "Valor": [
                "60 días",
                "[64]",
                "0.3",
                "0.001",
                "[32]",
                "150",
                "32"
            ]
        }
        
        config_df = pd.DataFrame(config_data)
        st.table(config_df)
    
    with col2:
        st.subheader("📊 Información del Dataset")
        
        df_original = load_dataset()
        
        if df_original is not None:
            dataset_info = {
                "Métrica": [
                    "Total de Observaciones",
                    "Período",
                    "Temperatura Promedio",
                    "Temperatura Máxima",
                    "Temperatura Mínima",
                    "Desviación Estándar"
                ],
                "Valor": [
                    f"{len(df_original)} días",
                    f"{df_original.index[0].strftime('%Y-%m-%d')} a {df_original.index[-1].strftime('%Y-%m-%d')}",
                    f"{df_original['Temp'].mean():.2f}°C",
                    f"{df_original['Temp'].max():.2f}°C",
                    f"{df_original['Temp'].min():.2f}°C",
                    f"{df_original['Temp'].std():.2f}°C"
                ]
            }
            
            dataset_df = pd.DataFrame(dataset_info)
            st.table(dataset_df)
    
    st.markdown("---")
    
    # Arquitectura del modelo
    st.subheader("🏗️ Arquitectura del Modelo")
    
    st.code("""
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    lstm (LSTM)                 (None, 64)                16896     
    dropout (Dropout)           (None, 64)                0         
    dense (Dense)               (None, 32)                2080      
    dropout_1 (Dropout)         (None, 32)                0         
    dense_1 (Dense)             (None, 1)                 33        
    =================================================================
    Total params: 19,009
    Trainable params: 19,009
    Non-trainable params: 0
    _________________________________________________________________
    """, language="text")
    
    st.markdown("---")
    
    # Detalles de entrenamiento
    st.subheader("📈 Detalles de Entrenamiento")
    
    training_details = """
    ### Proceso de Optimización
    
    1. **Grid Search** sobre hiperparámetros:
       - Sequence Length: [30, 60, 90]
       - LSTM Units: [[32], [64], [128]]
       - Dropout: [0.2, 0.3, 0.4]
       - Learning Rate: [0.001, 0.0001]
    
    2. **Validación:**
       - Train/Test Split: 80/20
       - Sin mezcla aleatoria (respeta orden temporal)
       - Early Stopping con patience=20
    
    3. **Optimizador:** Adam
    4. **Función de Pérdida:** MSE (Mean Squared Error)
    5. **Métricas:** MAE, RMSE, MAPE, R²
    """
    
    st.markdown(training_details)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🎓 <b>Proyecto Final - Modelos Predictivos</b></p>
    <p>Predicción de Temperaturas Mínimas con LSTM | Melbourne, Australia (1981-1990)</p>
</div>
""", unsafe_allow_html=True)
