"""
Módulo del Chatbot Asistente Inteligente para el Tasador y Buscador de Vehículos.

Características:
1. Extracción híbrida de parámetros (LLM Gemini con fallback automático offline mediante Regex).
2. Detección interactiva de parámetros faltantes (año, kilometraje, cilindraje).
3. Integración directa con los modelos de Machine Learning (Linear Regression + KNN).
4. Generación de avalúo comercial, tabla comparativa de similares y recomendación de mercado.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field

# Carga de variables de entorno
load_dotenv()

# Intentar importar el SDK de Google GenAI
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class VehicleExtractionSchema(BaseModel):
    """Esquema de salida estructurada JSON para Gemini."""
    year: Optional[int] = Field(
        default=None,
        description="Año del modelo del vehículo entre 1995 y 2026 (ej. 2018). Si no se menciona, es null.",
    )
    km_driven: Optional[float] = Field(
        default=None,
        description="Kilometraje recorrido en kilómetros (ej. 60000 para '60 mil km'). Si no se menciona, es null.",
    )
    engine_cc: Optional[float] = Field(
        default=None,
        description="Cilindraje en centímetros cúbicos (ej. 1600 para 1.6L o 1600 cc). Si no se menciona, es null.",
    )


@dataclass
class VehicleParameters:
    """Estructura de datos para parámetros de tasación."""
    year: Optional[int] = None
    km_driven: Optional[float] = None
    engine_cc: Optional[float] = None

    def is_complete(self) -> bool:
        """Verifica si todos los parámetros necesarios están presentes."""
        return (
            self.year is not None
            and self.km_driven is not None
            and self.engine_cc is not None
        )

    def missing_fields(self) -> List[str]:
        """Retorna lista legible de los campos faltantes."""
        missing = []
        if self.year is None:
            missing.append("Año del vehículo (ej. 2018)")
        if self.km_driven is None:
            missing.append("Kilometraje recorrido (ej. 60.000 km o 60 mil km)")
        if self.engine_cc is None:
            missing.append("Cilindraje del motor (ej. 1.6 o 1600 cc)")
        return missing

    def merge(self, other: "VehicleParameters") -> "VehicleParameters":
        """Combina parámetros existentes con nuevos datos de turnos conversacionales."""
        return VehicleParameters(
            year=other.year if other.year is not None else self.year,
            km_driven=other.km_driven if other.km_driven is not None else self.km_driven,
            engine_cc=other.engine_cc if other.engine_cc is not None else self.engine_cc,
        )


def get_gemini_client() -> Optional[Any]:
    """Inicializa el cliente de Gemini si la API key está disponible."""
    if not GENAI_AVAILABLE:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "tu_api_key_aqui"):
        return None
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception:
        return None


def extract_parameters_regex(text: str) -> VehicleParameters:
    """
    Extractor offline basado en expresiones regulares para detectar año, kilometraje y cilindraje.
    Maneja variaciones coloquiales en español (ej. '60 mil km', '60k', '1.6', '1600 cc', '2018').
    """
    text_lower = text.lower()
    engine_cc: Optional[float] = None
    km_driven: Optional[float] = None
    year: Optional[int] = None

    # 1. Extracción de Cilindraje (engine_cc)
    # Patrón explícito de CC (ej. 1600 cc, 2000cm3)
    m_cc = re.search(r"\b([6-9]\d{2}|[1-5]\d{3})\s*(?:cc|cm3)\b", text_lower)
    if m_cc:
        engine_cc = float(m_cc.group(1))
    else:
        # Patrón en litros: 1.6, 1.6L, 1.6 litros, 2.0
        m_liters = re.search(
            r"(?:motor|cilindraje|cilindrada)?\s*\b([0-5]\.[0-9])\s*(?:l|litros?|cc)?\b",
            text_lower,
        )
        if m_liters and m_liters.group(1):
            engine_cc = round(float(m_liters.group(1)) * 1000.0, 1)
        else:
            # 'motor 1600' o 'cilindraje 1400'
            m_motor = re.search(
                r"(?:motor|cilindraje|cilindrada)\s*(?:de)?\s*([6-9]\d{2}|[1-5]\d{3})\b",
                text_lower,
            )
            if m_motor:
                engine_cc = float(m_motor.group(1))

    # 2. Extracción de Kilometraje (km_driven)
    # Patrones con mil / k (ej. 60 mil km, 60 mil, 60k km, 60k)
    m_mil = re.search(
        r"\b(\d+(?:[\.,]\d+)?)\s*(?:mil|k)\s*(?:km|kms|kilometros?)?\b",
        text_lower,
    )
    if m_mil:
        val = float(m_mil.group(1).replace(",", "."))
        km_driven = val * 1000.0
    else:
        # Patrón numérico directo con separador de miles o unidad km (ej. 60.000 km, 60000 km)
        m_km = re.search(
            r"\b(\d{1,3}(?:\.\d{3})+|\d{4,7})\s*(?:km|kms|kilometros?)\b",
            text_lower,
        )
        if m_km:
            km_driven = float(m_km.group(1).replace(".", "").replace(",", ""))
        else:
            # Contexto explícito de kilometraje (ej. 'kilometraje: 60000')
            m_rec = re.search(
                r"(?:kilometraje|recorrido|con)\s*(?:de|es|:)?\s*(\d{4,7})\b",
                text_lower,
            )
            if m_rec:
                km_driven = float(m_rec.group(1))

    # 3. Extracción de Año (year entre 1995 y 2026)
    candidates = re.findall(r"\b(199[5-9]|20[0-2][0-9])\b", text_lower)
    for cand in candidates:
        cand_val = int(cand)
        # Evitar colisión si el número coincide con engine_cc especificado con 'cc'
        if engine_cc and abs(cand_val - engine_cc) < 1e-3 and "cc" in text_lower:
            continue
        year = cand_val
        break

    return VehicleParameters(year=year, km_driven=km_driven, engine_cc=engine_cc)


def extract_parameters_llm(
    text: str, client: Any
) -> Tuple[Optional[VehicleParameters], Optional[str]]:
    """
    Extrae parámetros utilizando el SDK google-genai con salida JSON estructurada.
    Intenta primero gemini-2.5-flash y ante retiro/indisponibilidad utiliza gemini-3.8-flash.
    """
    if client is None:
        return None, None

    prompt = (
        "Eres un analista de datos automotrices. Extrae del mensaje del usuario los siguientes campos:\n"
        "- year: Año del modelo (número entero entre 1995 y 2026, o null si no se menciona)\n"
        "- km_driven: Kilometraje en km (número, ej. 60000 para 60 mil km, o null si no se menciona)\n"
        "- engine_cc: Cilindraje en centímetros cúbicos (número, ej. 1600 para 1.6L o 1600 cc, o null)\n\n"
        f"Mensaje del usuario: \"{text}\""
    )

    models_to_try = ["gemini-2.5-flash", "gemini-3.8-flash"]

    for model_name in models_to_try:
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VehicleExtractionSchema,
                temperature=0.1,
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                data = json.loads(response.text)
                params = VehicleParameters(
                    year=data.get("year"),
                    km_driven=float(data["km_driven"]) if data.get("km_driven") is not None else None,
                    engine_cc=float(data["engine_cc"]) if data.get("engine_cc") is not None else None,
                )
                return params, f"LLM ({model_name})"
        except Exception:
            continue

    return None, None


def extract_parameters(
    user_message: str, client: Optional[Any] = None
) -> Tuple[VehicleParameters, str]:
    """
    Extractor híbrido:
    Intenta extraer con LLM Gemini; si no hay cliente o falla la llamada,
    activa automáticamente el extractor offline Regex.
    """
    # 1. Intento con LLM si el cliente está disponible
    if client is not None:
        params, method = extract_parameters_llm(user_message, client)
        if params is not None:
            return params, method

    # 2. Fallback automático offline con Regex
    params = extract_parameters_regex(user_message)
    return params, "Extractor Offline (Regex)"


def build_missing_info_response(missing: List[str]) -> str:
    """Genera una respuesta cordial solicitando los datos faltantes."""
    fields_list = "\n".join([f"- **{field}**" for field in missing])
    return (
        "¡Hola! Con gusto te ayudo a tasar tu vehículo y buscar las mejores opciones comparables en el mercado.\n\n"
        "Para calcular una estimación precisa, me falta conocer la siguiente información:\n"
        f"{fields_list}\n\n"
        "Por favor indícame estos datos para entregarte el avalúo completo."
    )


def build_valuation_response(
    year: int,
    km_driven: float,
    engine_cc: float,
    estimated_price: float,
    similar_df: pd.DataFrame,
    extraction_method: str,
) -> str:
    """
    Redacta una respuesta analítica y estructurada con el avalúo, resumen comparativo
    y recomendación estratégica de mercado.
    """
    # 1. Estadísticas de los vehículos similares (KNN)
    avg_similar_price = float(similar_df["price_cop"].mean())
    min_similar_price = float(similar_df["price_cop"].min())
    max_similar_price = float(similar_df["price_cop"].max())
    diff_cop = estimated_price - avg_similar_price
    pct_diff = ((estimated_price - avg_similar_price) / avg_similar_price) * 100.0

    # 2. Recomendación estratégica
    if pct_diff < -3.0:
        recomendacion = (
            f"**Oportunidad de Mercado (Precio Competitivo):** El avalúo estimado se ubica un "
            f"**{abs(pct_diff):.1f}% por debajo** del promedio de publicaciones similares en el mercado "
            f"(${avg_similar_price:,.0f} COP).\n"
            "- *Para Compradores:* Representa un precio altamente favorable y competitivo.\n"
            "- *Para Vendedores:* Si el vehículo está en óptimas condiciones, podrías ajustar tu oferta al alza "
            f"hacia los ${avg_similar_price:,.0f} COP."
        )
    elif pct_diff > 3.0:
        recomendacion = (
            f"**Precio Superior al Promedio:** El valor estimado está un **{pct_diff:.1f}% por encima** "
            f"del promedio de los modelos comparables (${avg_similar_price:,.0f} COP).\n"
            "- *Para Compradores:* Se recomienda verificar extras, historial mecánico y estado estético para justificar este diferencial.\n"
            "- *Para Vendedores:* Tu auto goza de buena valorización teórica; asegúrate de destacar su bajo desgaste y mantenimiento al publicarlo."
        )
    else:
        recomendacion = (
            f"**Precio en Equilibrio de Mercado:** El avalúo estimado (${estimated_price:,.0f} COP) está "
            f"alineado con el promedio de vehículos similares (${avg_similar_price:,.0f} COP), lo que "
            "indica un valor justo y equilibrado para una transacción rápida."
        )

    # 3. Construcción del resumen Markdown
    tabla_similares = []
    tabla_similares.append("| Vehículo | Año | Kilometraje | Cilindraje | Combustible | Transmisión | Precio Real | Similitud |")
    tabla_similares.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for _, row in similar_df.iterrows():
        nombre = row["name"]
        a = int(row["year"])
        km = f"{row['km_driven']:,.0f} km"
        cc = f"{row['engine_cc']:,.0f} cc"
        comb = row["fuel"]
        trans = row["transmission"]
        pr = f"${row['price_cop']:,.0f} COP"
        sim = f"{row['similarity_pct']:.1f}%"
        tabla_similares.append(f"| {nombre} | {a} | {km} | {cc} | {comb} | {trans} | {pr} | {sim} |")

    tabla_md = "\n".join(tabla_similares)

    response = (
        f"### 🚗 Resultado del Avalúo Comercial\n\n"
        f"- **Año:** `{year}`\n"
        "- **Kilometraje:** `" f"{km_driven:,.0f} km`\n"
        "- **Cilindraje:** `" f"{engine_cc:,.0f} cc`\n\n"
        f"### 💰 Precio Estimado de Mercado: **${estimated_price:,.0f} COP**\n\n"
        f"---\n\n"
        f"### 🔍 Top 5 Vehículos Más Similares del Mercado (KNN):\n"
        f"Rango de mercado de similares: **${min_similar_price:,.0f}** - **${max_similar_price:,.0f} COP** "
        f"(Promedio: **${avg_similar_price:,.0f} COP**).\n\n"
        f"{tabla_md}\n\n"
        f"---\n\n"
        f"### 💡 Análisis y Recomendación de Compra / Venta:\n"
        f"{recomendacion}\n\n"
        f"> 🤖 *Método de extracción utilizado: {extraction_method}*"
    )

    return response


def process_user_message(
    user_message: str,
    models_bundle: Dict[str, Any],
    df: pd.DataFrame,
    current_params: Optional[VehicleParameters] = None,
    client: Optional[Any] = None,
) -> Tuple[str, VehicleParameters, Optional[pd.DataFrame]]:
    """
    Función orquestadora del chatbot:
    1. Extrae parámetros del mensaje actual (híbrido LLM / Regex).
    2. Acumula/fusiona con parámetros previos de la sesión.
    3. Si faltan datos, solicita la información pendiente.
    4. Si los datos están completos, ejecuta la predicción ML y retorna el informe.
    """
    from src.models import predict_and_find_similar

    # 1. Extraer del mensaje actual
    new_params, method = extract_parameters(user_message, client=client)

    # 2. Fusionar con parámetros acumulados de la conversación
    if current_params is not None:
        accumulated_params = current_params.merge(new_params)
    else:
        accumulated_params = new_params

    # 3. Comprobar si faltan parámetros
    if not accumulated_params.is_complete():
        missing = accumulated_params.missing_fields()
        response_text = build_missing_info_response(missing)
        return response_text, accumulated_params, None

    # 4. Si están completos, llamar a los modelos de ML
    year = int(accumulated_params.year)
    km = float(accumulated_params.km_driven)
    cc = float(accumulated_params.engine_cc)

    pred_price, similar_df = predict_and_find_similar(
        year=year,
        km_driven=km,
        engine_cc=cc,
        models_bundle=models_bundle,
        df=df,
    )

    response_text = build_valuation_response(
        year=year,
        km_driven=km,
        engine_cc=cc,
        estimated_price=pred_price,
        similar_df=similar_df,
        extraction_method=method,
    )

    # Reiniciar parámetros tras completar la tasación exitosamente
    return response_text, VehicleParameters(), similar_df


def main():
    """Pruebas funcionales en consola del módulo de chatbot."""
    from src.models import train_and_evaluate

    data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "vehicles_unified.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    print("Cargando dataset y entrenando modelos...")
    df = pd.read_csv(data_path)
    bundle = train_and_evaluate(df)
    client = get_gemini_client()

    print(f"Cliente Gemini disponible: {client is not None}")
    print("=" * 60)
    print("SIMULACIÓN 1: Mensaje con datos incompletos")
    print("=" * 60)

    msg1 = "Hola, tengo un carro modelo 2018 con 60000 km, cuánto puede costar?"
    print(f"Usuario: {msg1}")
    resp1, state1, _ = process_user_message(msg1, bundle, df, client=client)
    print(f"\nAsistente:\n{resp1}\n")

    print("=" * 60)
    print("SIMULACIÓN 2: Completando el dato faltante en el siguiente turno")
    print("=" * 60)

    msg2 = "El motor es 1.6 litros"
    print(f"Usuario: {msg2}")
    resp2, state2, _ = process_user_message(msg2, bundle, df, current_params=state1, client=client)
    print(f"\nAsistente:\n{resp2}\n")

    print("=" * 60)
    print("SIMULACIÓN 3: Consulta completa en un solo mensaje")
    print("=" * 60)

    msg3 = "Cuánto vale un auto año 2016 con 45 mil km y motor 2000 cc?"
    print(f"Usuario: {msg3}")
    resp3, state3, _ = process_user_message(msg3, bundle, df, client=client)
    print(f"\nAsistente:\n{resp3}\n")


if __name__ == "__main__":
    main()
