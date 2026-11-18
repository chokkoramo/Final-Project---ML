"""
Aplicación Web: Tasador y Buscador de Vehículos Usados.
Interfaz optimizada, limpia y de alto rendimiento.
"""

from pathlib import Path
import os
import sys

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Módulos del proyecto
from src.chatbot import (
    VehicleParameters,
    get_gemini_client,
    process_user_message,
)
from src.models import (
    DEFAULT_MIN_PRICE_FLOOR,
    FEATURE_COLS,
    predict_and_find_similar,
    train_and_evaluate,
)

# Cargar variables de entorno
load_dotenv()

# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
st.set_page_config(
    page_title="Tasador de Vehículos Usados",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS sutil para mejorar legibilidad y espaciado
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            font-size: 1rem;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CACHING DE DATOS Y MODELOS
# ==============================================================================
@st.cache_data(show_spinner="Cargando dataset...")
def load_unified_data() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "processed" / "vehicles_unified.csv"
    if not data_path.exists():
        st.error(
            f"No se encontró el dataset en `{data_path}`. "
            "Ejecuta primero `src/etl.py`."
        )
        st.stop()
    return pd.read_csv(data_path)


@st.cache_resource(show_spinner="Entrenando modelos...")
def load_trained_models(_df: pd.DataFrame) -> dict:
    return train_and_evaluate(_df)


df_vehicles = load_unified_data()
models_bundle = load_trained_models(df_vehicles)

# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.title("AutoValuador Pro")
    st.caption("Tasación y recomendación vehicular basada en Machine Learning.")
    st.divider()

    st.markdown("**Modo de Inferencia**")
    mode_option = st.radio(
        "Selecciona el motor de procesamiento:",
        options=[
            "Híbrido (Fast-Path + LLM)",
            "Rápido (Local < 5 ms)",
            "LLM (Gemini 3.8 Flash)",
        ],
        index=0,
        label_visibility="collapsed",
        help="El modo híbrido analiza la consulta localmente en milisegundos y solo consulta a Gemini si es necesario.",
    )

    if "Híbrido" in mode_option:
        engine_mode = "hybrid"
    elif "Rápido" in mode_option:
        engine_mode = "fast"
    else:
        engine_mode = "llm"

    st.divider()
    st.markdown("**Configuración de API**")

    env_api_key = os.getenv("GEMINI_API_KEY", "")
    default_key = env_api_key if env_api_key and env_api_key != "tu_api_key_aqui" else ""

    user_api_key = st.text_input(
        "Gemini API Key:",
        value=default_key,
        type="password",
        placeholder="Opcional si usas modo local",
        help="Permite la interpretación con Gemini cuando se utiliza el modo LLM o consultas ambiguas.",
    )

    gemini_client = None
    if user_api_key.strip():
        gemini_client = get_gemini_client(api_key_override=user_api_key.strip())
        if gemini_client:
            st.success("Modo IA conectado (Gemini 3.8 Flash)")
        else:
            st.warning("Clave no válida. Operando en modo local.")
    else:
        st.info("Modo local activo")

    st.divider()
    st.markdown("**Resumen del Mercado**")
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Vehículos", f"{len(df_vehicles):,}")
    col_sb2.metric("Rango Años", f"{int(df_vehicles['year'].min())}-{int(df_vehicles['year'].max())}")
    st.caption(f"Precio promedio: ${df_vehicles['price_cop'].mean():,.0f} COP")

    st.divider()
    st.caption("Modelos: Regresión Lineal OLS + KNN con StandardScaler")

# ==============================================================================
# ENCABEZADO
# ==============================================================================
st.title("Tasador y Buscador de Vehículos Usados")
st.markdown(
    "Herramienta analítica para estimar el valor comercial de vehículos en Colombia, "
    "identificar modelos comparables en el mercado y realizar consultas en lenguaje natural."
)

# ==============================================================================
# PESTAÑAS PRINCIPALES
# ==============================================================================
tab_chat, tab_manual, tab_metrics = st.tabs(
    [
        "Asistente Virtual",
        "Tasador Manual",
        "Métricas del Modelo",
    ]
)

# ------------------------------------------------------------------------------
# PESTAÑA 1: ASISTENTE VIRTUAL (CHATBOT)
# ------------------------------------------------------------------------------
with tab_chat:
    st.markdown("### Asistente de Consulta")
    st.caption(
        "Indica las características del vehículo (año, kilometraje y cilindraje) "
        "para recibir la estimación y los vehículos más cercanos del mercado."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hola. Soy tu asesor de tasación vehicular en Colombia. "
                    "Indícame el **año**, **kilometraje** y **cilindraje** del vehículo a consultar "
                    "para calcular su avalúo y mostrarte los 5 modelos más parecidos en el mercado."
                ),
            }
        ]

    if "current_vehicle_params" not in st.session_state:
        st.session_state.current_vehicle_params = VehicleParameters()

    # Botones de consulta rápida
    st.markdown("**Consultas rápidas de prueba:**")
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.4, 0.8])

    quick_query = None
    with c1:
        if st.button("2018 | 60,000 km | 1.6L", use_container_width=True):
            quick_query = "Cuánto vale un carro 2018 con 60 mil km y motor 1.6?"
    with c2:
        if st.button("2016 | 45,000 km | 2.0L", use_container_width=True):
            quick_query = "Cuánto cuesta un auto año 2016 con 45000 km y motor 2000 cc?"
    with c3:
        if st.button("2019 | 80,000 km (Incompleto)", use_container_width=True):
            quick_query = "Tengo un vehículo 2019 con 80 mil km, cuánto podría costar?"
    with c4:
        if st.button("Limpiar chat", use_container_width=True):
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Historial reiniciado. ¿Qué vehículo deseas consultar?",
                }
            ]
            st.session_state.current_vehicle_params = VehicleParameters()
            st.rerun()

    # Historial de conversación
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada de texto
    user_prompt = st.chat_input("Escribe tu consulta (ej. Carro 2017 con 50 mil km y motor 1.4)...")
    if quick_query:
        user_prompt = quick_query

    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando consulta..."):
                response_text, updated_params, _ = process_user_message(
                    user_message=user_prompt,
                    models_bundle=models_bundle,
                    df=df_vehicles,
                    current_params=st.session_state.current_vehicle_params,
                    client=gemini_client,
                    mode=engine_mode,
                )
                st.session_state.current_vehicle_params = updated_params
                st.markdown(response_text)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": response_text}
        )

# ------------------------------------------------------------------------------
# PESTAÑA 2: TASADOR MANUAL Y BUSCADOR KNN
# ------------------------------------------------------------------------------
with tab_manual:
    st.markdown("### Tasador Paramétrico y Comparador KNN")
    st.caption("Ajusta los parámetros para calcular el avalúo y visualizar los 5 vehículos más cercanos.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        input_year = st.number_input(
            "Año de fabricación:",
            min_value=1995,
            max_value=2026,
            value=2018,
            step=1,
        )
    with col_in2:
        input_km = st.number_input(
            "Kilometraje recorrido (km):",
            min_value=0,
            max_value=500_000,
            value=60_000,
            step=5_000,
        )
    with col_in3:
        input_cc = st.number_input(
            "Cilindraje del motor (cc):",
            min_value=600,
            max_value=6_500,
            value=1_600,
            step=100,
        )

    btn_calc = st.button("Calcular avalúo", type="primary")

    if btn_calc or True:
        pred_price, similar_df = predict_and_find_similar(
            year=int(input_year),
            km_driven=float(input_km),
            engine_cc=float(input_cc),
            models_bundle=models_bundle,
            df=df_vehicles,
        )

        avg_similar_price = float(similar_df["price_cop"].mean())
        diff_price = pred_price - avg_similar_price
        pct_diff = ((pred_price - avg_similar_price) / avg_similar_price) * 100.0

        st.markdown("#### Resultados del Avalúo")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                label="Precio Estimado (Regresión Lineal)",
                value=f"${pred_price:,.0f} COP",
                delta=f"{pct_diff:+.1f}% vs promedio similares",
                delta_color="inverse" if pct_diff > 0 else "normal",
            )
        with m2:
            st.metric(
                label="Precio Promedio Similares (KNN)",
                value=f"${avg_similar_price:,.0f} COP",
            )
        with m3:
            st.metric(
                label="Diferencia frente al Mercado",
                value=f"${abs(diff_price):,.0f} COP",
                delta="Por encima" if diff_price > 0 else "Por debajo",
                delta_color="off",
            )

        # Diagnóstico de mercado
        if pct_diff < -3.0:
            st.info(
                f"**Oportunidad de mercado:** El precio estimado está un **{abs(pct_diff):.1f}% por debajo** "
                f"del promedio de vehículos similares (${avg_similar_price:,.0f} COP)."
            )
        elif pct_diff > 3.0:
            st.info(
                f"**Precio superior al promedio:** El precio estimado está un **{pct_diff:.1f}% por encima** "
                f"del promedio de vehículos similares (${avg_similar_price:,.0f} COP)."
            )
        else:
            st.info(
                f"**Precio en equilibrio:** El avalúo estimado está alineado con el promedio del mercado (${avg_similar_price:,.0f} COP)."
            )

        st.markdown("#### Vehículos más similares en el mercado (KNN)")

        display_df = similar_df.copy()
        display_df.rename(
            columns={
                "name": "Vehículo",
                "year": "Año",
                "km_driven": "Kilometraje",
                "engine_cc": "Cilindraje",
                "fuel": "Combustible",
                "transmission": "Transmisión",
                "price_cop": "Precio de Lista (COP)",
                "similarity_pct": "Similitud",
            },
            inplace=True,
        )
        display_df["Kilometraje"] = display_df["Kilometraje"].apply(lambda x: f"{x:,.0f} km")
        display_df["Cilindraje"] = display_df["Cilindraje"].apply(lambda x: f"{x:,.0f} cc")
        display_df["Precio de Lista (COP)"] = display_df["Precio de Lista (COP)"].apply(lambda x: f"${x:,.0f}")
        display_df["Similitud"] = display_df["Similitud"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_df[
                [
                    "Vehículo",
                    "Año",
                    "Kilometraje",
                    "Cilindraje",
                    "Combustible",
                    "Transmisión",
                    "Precio de Lista (COP)",
                    "Similitud",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------------------------------
# PESTAÑA 3: MÉTRICAS Y ANÁLISIS DEL MODELO
# ------------------------------------------------------------------------------
with tab_metrics:
    st.markdown("### Rendimiento y Diagnóstico del Modelo")
    st.caption("Evaluación estadística sobre el conjunto de prueba (20% de los datos).")

    metrics = models_bundle["metrics"]
    coefs = metrics["coefficients"]

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(
            label="Coeficiente de Determinación (R²)",
            value=f"{metrics['r2']:.4f}",
            help="Proporción de la varianza del precio explicada por el modelo (73.16%).",
        )
    with col_m2:
        st.metric(
            label="Error Absoluto Medio (MAE)",
            value=f"${metrics['mae']:,.0f} COP",
            help="Desviación promedio absoluta entre el precio real y la predicción.",
        )
    with col_m3:
        st.metric(
            label="Raíz del Error Cuadrático Medio (RMSE)",
            value=f"${metrics['rmse']:,.0f} COP",
            help="Penaliza desviaciones mayores en las estimaciones.",
        )

    st.markdown("#### Ecuación de Regresión Lineal")
    st.latex(
        r"\text{Precio (COP)} = \beta_0 + \beta_1 \cdot \text{Año} + \beta_2 \cdot \text{Km} + \beta_3 \cdot \text{Cilindraje}"
    )

    st.markdown(
        f"""
        - **Intercepto ($\beta_0$):** `{metrics['intercept']:,.2f}`
        - **Año ($\beta_1$):** `+{coefs['year']:,.2f} COP/año` $\\rightarrow$ Cada año más reciente añade en promedio **~${coefs['year']/1e6:.2f}M COP**.
        - **Kilometraje ($\beta_2$):** `{coefs['km_driven']:,.2f} COP/km` $\\rightarrow$ Cada kilómetro recorrido reduce el valor en **${abs(coefs['km_driven']):.2f} COP**.
        - **Cilindraje ($\beta_3$):** `+{coefs['engine_cc']:,.2f} COP/cc` $\\rightarrow$ Cada 100 cc adicionales incrementan el valor en **~${(coefs['engine_cc']*100)/1e6:.2f}M COP**.
        """
    )

    st.markdown("#### Gráficos de Diagnóstico")
    test_data = models_bundle["test_data"]
    y_test = test_data["y_test"]
    y_pred = test_data["y_pred"]
    X_test = test_data["X_test"]

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.2), dpi=100)
        ax1.scatter(
            y_test / 1e6,
            y_pred / 1e6,
            alpha=0.25,
            color="#2563eb",
            edgecolors="none",
            s=18,
            label="Datos de prueba",
        )
        max_val = max(y_test.max(), y_pred.max()) / 1e6
        ax1.plot([0, max_val], [0, max_val], color="#dc2626", linestyle="--", linewidth=1.8, label="Línea ideal (y = x)")
        ax1.set_title("Precio Real vs. Precio Predicho", fontsize=11, fontweight="600")
        ax1.set_xlabel("Precio Real (Millones COP)", fontsize=9)
        ax1.set_ylabel("Precio Predicho (Millones COP)", fontsize=9)
        ax1.set_xlim(0, 300)
        ax1.set_ylim(0, 300)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left", frameon=True, fontsize=8)
        st.pyplot(fig1)
        plt.close(fig1)

    with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.2), dpi=100)
        ax2.scatter(
            X_test["year"],
            y_test / 1e6,
            alpha=0.20,
            color="#059669",
            edgecolors="none",
            s=18,
            label="Precios reales",
        )
        avg_by_year = (
            pd.DataFrame({"year": X_test["year"], "price_cop": y_test})
            .groupby("year")["price_cop"]
            .mean()
            / 1e6
        )
        ax2.plot(
            avg_by_year.index,
            avg_by_year.values,
            color="#d97706",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Promedio por año",
        )
        ax2.set_title("Relación Año de Fabricación vs. Precio", fontsize=11, fontweight="600")
        ax2.set_xlabel("Año de Fabricación", fontsize=9)
        ax2.set_ylabel("Precio (Millones COP)", fontsize=9)
        ax2.set_ylim(0, 300)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="upper left", frameon=True, fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.divider()
st.caption("Proyecto de Machine Learning: Regresión Lineal, K-Nearest Neighbors y Asistente Virtual.")
