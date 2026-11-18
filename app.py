"""
Aplicación Web Principal: Tasador y Buscador de Vehículos Usados.
Desarrollada con Streamlit, Scikit-Learn y Google GenAI.
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

# Importar módulos del proyecto
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
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Tasador y Buscador de Vehículos Usados",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CACHING DE DATOS Y MODELOS
# ==============================================================================
@st.cache_data(show_spinner="Cargando dataset de vehículos unificado...")
def load_unified_data() -> pd.DataFrame:
    """Carga y almacena en caché el dataset procesado."""
    data_path = PROJECT_ROOT / "data" / "processed" / "vehicles_unified.csv"
    if not data_path.exists():
        st.error(
            f"No se encontró el dataset procesado en `{data_path}`. "
            "Por favor ejecuta primero `src/etl.py`."
        )
        st.stop()
    return pd.read_csv(data_path)


@st.cache_resource(show_spinner="Entrenando modelos de Regresión Lineal y KNN...")
def load_trained_models(_df: pd.DataFrame) -> dict:
    """Entrena y almacena en caché los modelos de Machine Learning."""
    return train_and_evaluate(_df)


# Cargar recursos cacheados
df_vehicles = load_unified_data()
models_bundle = load_trained_models(df_vehicles)

# ==============================================================================
# SIDEBAR (BARRA LATERAL)
# ==============================================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=600&q=80",
        use_container_width=True,
    )
    st.title("🚗 AutoValuador Pro")
    st.markdown(
        "**Tasador Inteligente de Vehículos Usados** desarrollado con "
        "modelos de Machine Learning y Asistente Virtual."
    )
    st.markdown("---")

    st.subheader("⚡ Optimización del Chatbot")

    # Selector de Motor de Inferencia
    mode_option = st.radio(
        "Modo de Velocidad / Inferencia:",
        options=[
            "🚀 Híbrido Inteligente (Recomendado)",
            "⚡ Modo Turbo (Ultra Rápido < 5 ms)",
            "🤖 Modo LLM Completo (Gemini)",
        ],
        index=0,
        help=(
            "• Híbrido: Extrae con Fast-Path local instantáneo (<5 ms) y usa Gemini para consultas complejas.\n"
            "• Turbo: 100% procesamiento local inmediato sin llamadas de red.\n"
            "• LLM Completo: Envía siempre la consulta a la nube de Google."
        ),
    )

    if "Híbrido" in mode_option:
        engine_mode = "hybrid"
    elif "Turbo" in mode_option:
        engine_mode = "fast"
    else:
        engine_mode = "llm"

    st.markdown("---")
    st.subheader("🔑 Configuración de API")

    env_api_key = os.getenv("GEMINI_API_KEY", "")
    default_key = env_api_key if env_api_key and env_api_key != "tu_api_key_aqui" else ""

    user_api_key = st.text_input(
        "Gemini API Key (opcional):",
        value=default_key,
        type="password",
        help="Si ingresas tu API Key, el chatbot usará Gemini con salida estructurada JSON y respuestas conversacionales.",
    )

    gemini_client = None
    if user_api_key.strip():
        gemini_client = get_gemini_client(api_key_override=user_api_key.strip())
        if gemini_client:
            st.success("🟢 **Modo IA Conectado** (Gemini 3.8 Flash)")
        else:
            st.warning("⚠️ Clave no válida. Operando en modo local.")
    else:
        st.info("🟡 **Modo Local Activo** (Sin llamadas externas)")

    st.markdown("---")
    st.subheader("📊 Catálogo del Mercado")
    st.markdown(f"- **Total Vehículos:** `{len(df_vehicles):,}`")
    st.markdown(f"- **Rango Años:** `{int(df_vehicles['year'].min())}` - `{int(df_vehicles['year'].max())}`")
    st.markdown(
        f"- **Precio Promedio:** `${df_vehicles['price_cop'].mean():,.0f} COP`"
    )
    st.markdown(
        f"- **Marcas / Modelos:** `{df_vehicles['name'].nunique():,}` referencias"
    )

    st.markdown("---")
    st.caption("Algoritmos: Regresión Lineal OLS + K-Nearest Neighbors (KNN)")

# ==============================================================================
# ENCABEZADO PRINCIPAL
# ==============================================================================
st.title("🚗 Tasador y Buscador de Vehículos Usados")
st.markdown(
    "Estima el valor comercial de cualquier vehículo en Colombia mediante "
    "**Regresión Lineal**, encuentra los 5 vehículos más similares del mercado con **KNN** "
    "y consulta interactivamente con nuestro **Asistente Virtual**."
)

# ==============================================================================
# PESTAÑAS PRINCIPALES
# ==============================================================================
tab_chat, tab_manual, tab_metrics = st.tabs(
    [
        "💬 Asistente Virtual (Chatbot)",
        "🔍 Tasador Manual y Buscador KNN",
        "📈 Análisis y Métricas del Modelo",
    ]
)

# ------------------------------------------------------------------------------
# PESTAÑA 1: CHATBOT ASISTENTE
# ------------------------------------------------------------------------------
with tab_chat:
    st.subheader("💬 Consulta en Lenguaje Natural con el Asistente")
    st.markdown(
        "Pregúntale al chatbot sobre cualquier auto. Puedes escribir oraciones como: "
        "*'Tengo un carro 2018 con 60 mil km y motor 1.6, cuánto vale?'* o interactuar turno a turno."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! 👋 Soy tu asesor inteligente de vehículos usados en Colombia. "
                    "Indícame el **año**, **kilometraje** y **cilindraje** del vehículo que deseas consultar "
                    "y calcularé su valor comercial estimado junto a las 5 mejores referencias del mercado."
                ),
            }
        ]

    if "current_vehicle_params" not in st.session_state:
        st.session_state.current_vehicle_params = VehicleParameters()

    st.markdown("**Consultas rápidas de ejemplo (clic para probar instantáneamente):**")
    col_q1, col_q2, col_q3, col_clear = st.columns([1.2, 1.2, 1.4, 0.8])

    quick_query = None
    with col_q1:
        if st.button("🚗 2018, 60k km, 1.6L", use_container_width=True):
            quick_query = "Cuánto vale un carro 2018 con 60 mil km y motor 1.6?"
    with col_q2:
        if st.button("⚡ 2016, 45k km, 2.0L", use_container_width=True):
            quick_query = "Cuánto cuesta un auto año 2016 con 45000 km y motor 2000 cc?"
    with col_q3:
        if st.button("❓ 2019, 80k km (sin motor)", use_container_width=True):
            quick_query = "Tengo un vehículo 2019 con 80 mil km, cuánto podría costar?"
    with col_clear:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "Historial reiniciado. ¿Qué vehículo deseas tasar?",
                }
            ]
            st.session_state.current_vehicle_params = VehicleParameters()
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Escribe tu consulta aquí (ej. Carro 2017 con 50 mil km y motor 1.4)...")
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
    st.subheader("🔍 Tasador Paramétrico y Recomendador de Similares")
    st.markdown(
        "Ajusta las características del vehículo para obtener inmediatamente su tasación "
        "con **Regresión Lineal** y la lista de los 5 vehículos más cercanos calculada con **K-Nearest Neighbors (KNN)**."
    )

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        input_year = st.number_input(
            "📅 Año de Fabricación:",
            min_value=1995,
            max_value=2026,
            value=2018,
            step=1,
            help="Año del modelo del vehículo.",
        )
    with col_in2:
        input_km = st.number_input(
            "🛣️ Kilometraje Recorrido (km):",
            min_value=0,
            max_value=500_000,
            value=60_000,
            step=5_000,
            help="Total de kilómetros recorridos por el automóvil.",
        )
    with col_in3:
        input_cc = st.number_input(
            "⚙️ Cilindraje del Motor (cc):",
            min_value=600,
            max_value=6_500,
            value=1_600,
            step=100,
            help="Capacidad del motor en centímetros cúbicos (ej. 1600 para 1.6L).",
        )

    btn_tasar = st.button("🚀 Calcular Avalúo y Buscar Similares", type="primary", use_container_width=True)

    if btn_tasar or True:
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

        st.markdown("### 📊 Resultados de la Valoración")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            st.metric(
                label="💰 Precio Estimado (Regresión Lineal)",
                value=f"${pred_price:,.0f} COP",
                delta=f"{pct_diff:+.1f}% vs promedio similares",
                delta_color="inverse" if pct_diff > 0 else "normal",
            )
        with m_col2:
            st.metric(
                label="👥 Precio Promedio de 5 Similares (KNN)",
                value=f"${avg_similar_price:,.0f} COP",
            )
        with m_col3:
            st.metric(
                label="⚖️ Diferencia vs Mercado",
                value=f"${abs(diff_price):,.0f} COP",
                delta="Por encima" if diff_price > 0 else "Por debajo",
                delta_color="off",
            )

        if pct_diff < -3.0:
            st.success(
                f"💡 **Oportunidad de Compra:** El precio estimado está un **{abs(pct_diff):.1f}% por debajo** "
                f"del promedio de vehículos similares del catálogo (${avg_similar_price:,.0f} COP)."
            )
        elif pct_diff > 3.0:
            st.warning(
                f"⚠️ **Precio Superior a la Media:** El precio estimado se sitúa un **{pct_diff:.1f}% por encima** "
                f"de los modelos similares (${avg_similar_price:,.0f} COP). Revisa detalles de equipamiento y estado mecánico."
            )
        else:
            st.info(
                f"✅ **Precio en Equilibrio:** El avalúo está prácticamente a la par del promedio del mercado (${avg_similar_price:,.0f} COP)."
            )

        st.markdown("#### 🚗 Top 5 Vehículos Más Similares del Catálogo (KNN con StandardScaler)")

        display_df = similar_df.copy()
        display_df.rename(
            columns={
                "name": "Vehículo",
                "year": "Año",
                "km_driven": "Kilometraje (km)",
                "engine_cc": "Cilindraje (cc)",
                "fuel": "Combustible",
                "transmission": "Transmisión",
                "price_cop": "Precio Real (COP)",
                "similarity_pct": "Similitud (%)",
            },
            inplace=True,
        )
        display_df["Kilometraje (km)"] = display_df["Kilometraje (km)"].apply(lambda x: f"{x:,.0f} km")
        display_df["Cilindraje (cc)"] = display_df["Cilindraje (cc)"].apply(lambda x: f"{x:,.0f} cc")
        display_df["Precio Real (COP)"] = display_df["Precio Real (COP)"].apply(lambda x: f"${x:,.0f}")
        display_df["Similitud (%)"] = display_df["Similitud (%)"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_df[
                [
                    "Vehículo",
                    "Año",
                    "Kilometraje (km)",
                    "Cilindraje (cc)",
                    "Combustible",
                    "Transmisión",
                    "Precio Real (COP)",
                    "Similitud (%)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------------------------------
# PESTAÑA 3: ANÁLISIS Y MÉTRICAS DEL MODELO
# ------------------------------------------------------------------------------
with tab_metrics:
    st.subheader("📈 Rendimiento del Modelo y Sustentación Técnica")
    st.markdown(
        "Métricas cuantitativas obtenidas sobre el conjunto de prueba (20% de los datos) "
        "y descomposición de la ecuación de Regresión Lineal aprendida."
    )

    metrics = models_bundle["metrics"]
    coefs = metrics["coefficients"]

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(
            label="Coeficiente de Determinación (R²)",
            value=f"{metrics['r2']:.4f}",
            help="Porcentaje de la varianza del precio comercial explicada por el modelo (73.16%).",
        )
    with col_m2:
        st.metric(
            label="Error Absoluto Medio (MAE)",
            value=f"${metrics['mae']:,.0f} COP",
            help="Desviación promedio absoluta entre el precio real y el predicho.",
        )
    with col_m3:
        st.metric(
            label="Raíz Error Cuadrático Medio (RMSE)",
            value=f"${metrics['rmse']:,.0f} COP",
            help="Métrica que penaliza desvíos grandes en la estimación.",
        )

    st.markdown("#### 📐 Ecuación Aprendida de Regresión Lineal")
    st.latex(
        r"\text{Precio (COP)} = \beta_0 + \beta_1 \cdot \text{Año} + \beta_2 \cdot \text{Km} + \beta_3 \cdot \text{Cilindraje}"
    )

    st.markdown(
        f"""
        - **Intercepto ($\beta_0$):** `{metrics['intercept']:,.2f}`
        - **$\beta_1$ (Año):** `+{coefs['year']:,.2f} COP/año` $\\rightarrow$ Cada año más reciente añade en promedio **~${coefs['year']/1e6:.2f}M COP**.
        - **$\beta_2$ (Kilometraje):** `{coefs['km_driven']:,.2f} COP/km` $\\rightarrow$ Cada kilómetro recorrido deprecia el auto en **${abs(coefs['km_driven']):.2f} COP**.
        - **$\beta_3$ (Cilindraje):** `+{coefs['engine_cc']:,.2f} COP/cc` $\\rightarrow$ Cada 100 cc adicionales incrementan el valor en **~${(coefs['engine_cc']*100)/1e6:.2f}M COP**.
        """
    )

    st.markdown("#### 📊 Visualizaciones de Diagnóstico")
    test_data = models_bundle["test_data"]
    y_test = test_data["y_test"]
    y_pred = test_data["y_pred"]
    X_test = test_data["X_test"]

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=100)
        ax1.scatter(
            y_test / 1e6,
            y_pred / 1e6,
            alpha=0.25,
            color="#1f77b4",
            edgecolors="none",
            s=20,
            label="Vehículos de Prueba (Test)",
        )
        max_val = max(y_test.max(), y_pred.max()) / 1e6
        ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Línea Ideal (y = x)")
        ax1.set_title("Precio Real vs. Precio Predicho", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Precio Real (Millones COP)", fontsize=10)
        ax1.set_ylabel("Precio Predicho (Millones COP)", fontsize=10)
        ax1.set_xlim(0, 300)
        ax1.set_ylim(0, 300)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left")
        st.pyplot(fig1)
        plt.close(fig1)

    with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=100)
        ax2.scatter(
            X_test["year"],
            y_test / 1e6,
            alpha=0.25,
            color="#2ca02c",
            edgecolors="none",
            s=20,
            label="Precios Reales",
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
            color="#d62728",
            linewidth=2.5,
            marker="o",
            label="Tendencia Promedio",
        )
        ax2.set_title("Relación Año de Fabricación vs. Precio", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Año de Fabricación", fontsize=10)
        ax2.set_ylabel("Precio (Millones COP)", fontsize=10)
        ax2.set_ylim(0, 300)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left")
        st.pyplot(fig2)
        plt.close(fig2)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<center><small>Proyecto Final de Machine Learning | Universidad / Curso ML | "
    "Desarrollado con Streamlit, Scikit-Learn y Google GenAI</small></center>",
    unsafe_allow_html=True,
)
