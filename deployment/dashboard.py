"""
Dashboard interactivo para predicción de incendios forestales.
Aplicación web con Streamlit para visualización y predicción en tiempo real.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.predict import FirePredictionSystem

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Incendios Forestales",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prediction_system():
    """Carga el sistema de predicción (con caché)."""
    model_path = './models/random_forest_model.pkl'
    scaler_path = './models/scaler.pkl'
    
    if not Path(model_path).exists():
        return None
    
    try:
        predictor = FirePredictionSystem(
            model_path=model_path,
            scaler_path=scaler_path if Path(scaler_path).exists() else None
        )
        return predictor
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


def get_risk_color(risk_level: str) -> str:
    """Retorna el color según el nivel de riesgo."""
    colors = {
        'BAJO': '#28a745',
        'MEDIO': '#ffc107',
        'ALTO': '#fd7e14',
        'CRÍTICO': '#dc3545'
    }
    return colors.get(risk_level, '#6c757d')


def create_gauge_chart(value: float, title: str, max_value: float = 100) -> go.Figure:
    """Crea un gráfico de tipo gauge (medidor)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value * 0.33], 'color': "lightgreen"},
                {'range': [max_value * 0.33, max_value * 0.66], 'color': "yellow"},
                {'range': [max_value * 0.66, max_value], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def prediction_page():
    """Página de predicción individual."""
    st.title("🔥 Predicción de Incendios Forestales")
    st.markdown("---")
    
    # Cargar sistema de predicción
    predictor = load_prediction_system()
    
    if predictor is None:
        st.error("⚠️ No se pudo cargar el modelo de predicción. Verifica que el archivo exista.")
        st.info("📝 Entrena un modelo primero usando `train_random_forest.py`")
        return
    
    st.success("✓ Modelo cargado correctamente")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        Este sistema utiliza Machine Learning para predecir 
        el riesgo de incendios forestales basándose en 
        condiciones meteorológicas y ambientales.
        
        **Niveles de riesgo:**
        - 🟢 Sin Riesgo
        - 🟡 Riesgo Bajo
        - 🟠 Riesgo Medio
        - 🔴 Riesgo Alto
        - ⛔ Riesgo Crítico
        """)
        
        st.markdown("---")
        st.markdown("**Modelo:** Random Forest")
        st.markdown(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Formulario de entrada
    st.header("📊 Ingrese los datos meteorológicos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperatura = st.slider(
            "🌡️ Temperatura (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=25.0,
            step=0.5,
            help="Temperatura ambiente en grados Celsius"
        )
        
        humedad = st.slider(
            "💧 Humedad (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Humedad relativa del aire"
        )
        
        velocidad_viento = st.slider(
            "💨 Velocidad del viento (km/h)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Velocidad del viento"
        )
    
    with col2:
        precipitacion = st.slider(
            "🌧️ Precipitación (mm)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            help="Precipitación acumulada"
        )
        
        indice_sequedad = st.slider(
            "🌵 Índice de sequedad",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Índice de sequedad del suelo y vegetación"
        )
        
        vegetacion_seca = st.slider(
            "🍂 Vegetación seca (proporción)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Proporción de vegetación seca (0-1)"
        )
    
    with col3:
        mes = st.selectbox(
            "📅 Mes",
            options=list(range(1, 13)),
            index=datetime.now().month - 1,
            format_func=lambda x: datetime(2000, x, 1).strftime('%B')
        )
        
        dia_semana = st.selectbox(
            "📆 Día de la semana",
            options=list(range(7)),
            index=datetime.now().weekday(),
            format_func=lambda x: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 
                                   'Viernes', 'Sábado', 'Domingo'][x]
        )
        
        st.markdown("---")
        predict_button = st.button("🔮 Realizar Predicción", type="primary", use_container_width=True)
    
    # Mostrar gauges con datos de entrada
    st.markdown("---")
    st.subheader("📈 Visualización de Variables")
    
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    
    with gauge_col1:
        st.plotly_chart(
            create_gauge_chart(temperatura, "Temperatura (°C)", 50),
            use_container_width=True
        )
    
    with gauge_col2:
        st.plotly_chart(
            create_gauge_chart(humedad, "Humedad (%)", 100),
            use_container_width=True
        )
    
    with gauge_col3:
        st.plotly_chart(
            create_gauge_chart(velocidad_viento, "Viento (km/h)", 100),
            use_container_width=True
        )
    
    # Realizar predicción
    if predict_button:
        with st.spinner("Realizando predicción..."):
            # Preparar datos
            input_data = {
                'temperatura': temperatura,
                'humedad': humedad,
                'velocidad_viento': velocidad_viento,
                'precipitacion': precipitacion,
                'indice_sequedad': indice_sequedad,
                'vegetacion_seca': vegetacion_seca,
                'mes': mes,
                'dia_semana': dia_semana
            }
            
            # Realizar predicción
            result = predictor.predict_with_details(input_data)
            
            # Mostrar resultados
            st.markdown("---")
            st.header("🎯 Resultado de la Predicción")
            
            # Tarjeta de resultado principal
            risk_color = get_risk_color(result['risk_level'])
            
            st.markdown(f"""
                <div style="background-color: {risk_color}; padding: 2rem; border-radius: 1rem; 
                            text-align: center; color: white; margin: 1rem 0;">
                    <h1 style="margin: 0; font-size: 3rem;">
                        {result['prediction_label']}
                    </h1>
                    <h3 style="margin: 0.5rem 0 0 0;">
                        Nivel de Riesgo: {result['risk_level']}
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Métricas adicionales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['confidence']:
                    st.metric(
                        "Confianza de Predicción",
                        f"{result['confidence']:.1%}",
                        delta=None
                    )
            
            with col2:
                st.metric(
                    "Clase Predicha",
                    result['prediction'],
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Timestamp",
                    datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S'),
                    delta=None
                )
            
            # Probabilidades por clase
            if result['probabilities']:
                st.markdown("---")
                st.subheader("📊 Distribución de Probabilidades")
                
                prob_df = pd.DataFrame({
                    'Clase': ['Sin Riesgo', 'Riesgo Bajo', 'Riesgo Medio', 
                             'Riesgo Alto', 'Riesgo Crítico'][:len(result['probabilities'])],
                    'Probabilidad': result['probabilities']
                })
                
                fig = px.bar(
                    prob_df,
                    x='Clase',
                    y='Probabilidad',
                    color='Probabilidad',
                    color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                    labels={'Probabilidad': 'Probabilidad'},
                    text='Probabilidad'
                )
                
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Clase de Riesgo",
                    yaxis_title="Probabilidad",
                    yaxis_tickformat='.0%'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            st.markdown("---")
            st.subheader("💡 Recomendaciones")
            
            if result['risk_level'] in ['CRÍTICO', 'ALTO']:
                st.error("""
                ⚠️ **ALERTA DE ALTO RIESGO**
                
                - Evitar actividades que puedan generar chispas
                - Mantener vigilancia constante
                - Tener equipos de extinción preparados
                - Notificar a autoridades competentes
                - Restringir acceso a zonas forestales
                """)
            elif result['risk_level'] == 'MEDIO':
                st.warning("""
                ⚡ **PRECAUCIÓN**
                
                - Extremar medidas de prevención
                - Monitorear condiciones meteorológicas
                - Preparar planes de contingencia
                - Mantener comunicación con autoridades
                """)
            else:
                st.info("""
                ✓ **CONDICIONES FAVORABLES**
                
                - Mantener prácticas de prevención habituales
                - Continuar monitoreo de condiciones
                - Seguir protocolos establecidos
                """)


def batch_prediction_page():
    """Página de predicción por lotes."""
    st.title("📦 Predicción por Lotes")
    st.markdown("---")
    
    # Cargar sistema de predicción
    predictor = load_prediction_system()
    
    if predictor is None:
        st.error("⚠️ No se pudo cargar el modelo de predicción.")
        return
    
    st.info("""
    📝 Sube un archivo CSV con las siguientes columnas:
    - temperatura
    - humedad
    - velocidad_viento
    - precipitacion
    - indice_sequedad
    - vegetacion_seca
    - mes
    - dia_semana
    """)
    
    # Upload de archivo
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV",
        type=['csv'],
        help="El archivo debe contener las columnas mencionadas arriba"
    )
    
    if uploaded_file is not None:
        try:
            # Leer archivo
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ Archivo cargado: {len(df)} observaciones")
            
            # Mostrar preview
            st.subheader("👀 Vista Previa de Datos")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Botón de predicción
            if st.button("🔮 Realizar Predicciones", type="primary"):
                with st.spinner("Procesando predicciones..."):
                    # Realizar predicciones
                    results_df = predictor.batch_predict(df)
                    
                    st.success(f"✓ {len(results_df)} predicciones completadas")
                    
                    # Mostrar resultados
                    st.markdown("---")
                    st.subheader("📊 Resultados")
                    
                    # Métricas generales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total de Predicciones",
                            len(results_df)
                        )
                    
                    with col2:
                        high_risk = len(results_df[results_df['prediccion'] >= 3])
                        st.metric(
                            "Alto Riesgo",
                            high_risk,
                            delta=f"{high_risk/len(results_df)*100:.1f}%"
                        )
                    
                    with col3:
                        medium_risk = len(results_df[results_df['prediccion'] == 2])
                        st.metric(
                            "Riesgo Medio",
                            medium_risk,
                            delta=f"{medium_risk/len(results_df)*100:.1f}%"
                        )
                    
                    with col4:
                        low_risk = len(results_df[results_df['prediccion'] <= 1])
                        st.metric(
                            "Bajo Riesgo",
                            low_risk,
                            delta=f"{low_risk/len(results_df)*100:.1f}%"
                        )
                    
                    # Gráfico de distribución
                    st.markdown("---")
                    st.subheader("📈 Distribución de Predicciones")
                    
                    pred_counts = results_df['etiqueta_prediccion'].value_counts()
                    
                    fig = px.pie(
                        values=pred_counts.values,
                        names=pred_counts.index,
                        title="Distribución de Niveles de Riesgo",
                        color_discrete_sequence=px.colors.sequential.RdYlGn_r
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de resultados
                    st.markdown("---")
                    st.subheader("📋 Tabla de Resultados")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Descargar resultados
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Descargar Resultados (CSV)",
                        data=csv,
                        file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")


def statistics_page():
    """Página de estadísticas del sistema."""
    st.title("📊 Estadísticas del Sistema")
    st.markdown("---")
    
    # Cargar sistema de predicción
    predictor = load_prediction_system()
    
    if predictor is None:
        st.error("⚠️ No se pudo cargar el modelo de predicción.")
        return
    
    # Obtener estadísticas
    stats = predictor.get_statistics()
    
    if stats['total_predictions'] == 0:
        st.info("📝 No hay predicciones registradas todavía.")
        return
    
    # Mostrar estadísticas generales
    st.subheader("📈 Estadísticas Generales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total de Predicciones",
            stats['total_predictions']
        )
    
    with col2:
        if stats['prediction_distribution']:
            most_common = max(stats['prediction_distribution'].items(), 
                            key=lambda x: x[1])
            st.metric(
                "Predicción Más Común",
                most_common[0],
                delta=f"{most_common[1]} veces"
            )
    
    with col3:
        if stats['most_common_prediction'] is not None:
            st.metric(
                "Clase Modal",
                stats['most_common_prediction']
            )
    
    # Distribución de predicciones
    if stats['prediction_distribution']:
        st.markdown("---")
        st.subheader("📊 Distribución de Predicciones")
        
        dist_df = pd.DataFrame(
            list(stats['prediction_distribution'].items()),
            columns=['Predicción', 'Frecuencia']
        )
        
        fig = px.bar(
            dist_df,
            x='Predicción',
            y='Frecuencia',
            title="Frecuencia de Predicciones",
            color='Frecuencia',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Función principal de la aplicación."""
    
    # Sidebar de navegación
    st.sidebar.title("🧭 Navegación")
    page = st.sidebar.radio(
        "Selecciona una página:",
        ["🔮 Predicción Individual", "📦 Predicción por Lotes", "📊 Estadísticas"]
    )
    
    # Navegación entre páginas
    if page == "🔮 Predicción Individual":
        prediction_page()
    elif page == "📦 Predicción por Lotes":
        batch_prediction_page()
    elif page == "📊 Estadísticas":
        statistics_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Desarrollado por:** Herwig
    
    **Tecnologías:** 
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    """)


if __name__ == "__main__":
    main()
