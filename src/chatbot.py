"""
Módulo del Chatbot Asistente Inteligente de Alto Rendimiento para el Tasador de Vehículos.

Optimizaciones de Rendimiento:
1. Fast-Path First: Extracción local con expresiones regulares en ~2 ms antes de invocar la red.
2. Eliminación de reintentos en modelos deprecados (llamada directa a gemini-3.8-flash).
3. Timeout estricto asíncrono (máximo 3.0 segundos) para llamadas LLM, evitando cuelgues por saturación de Google.
4. Soporte para consultas conversacionales generales y preguntas frecuentes.
"""

import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
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

    def has_any(self) -> bool:
        """Verifica si al menos un parámetro fue detectado."""
        return (
            self.year is not None
            or self.km_driven is not None
            or self.engine_cc is not None
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


def get_gemini_client(api_key_override: Optional[str] = None) -> Optional[Any]:
    """Inicializa el cliente de Gemini si la API key está disponible."""
    if not GENAI_AVAILABLE:
        return None
    api_key = api_key_override or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "tu_api_key_aqui"):
        return None
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception:
        return None


def extract_parameters_regex(text: str) -> VehicleParameters:
    """
    Extractor local ultrarrápido basado en expresiones regulares (latencia típica: ~1-3 ms).
    Maneja variaciones en español ('60 mil km', '60k', '1.6', '1600 cc', '2018', etc.).
    """
    text_lower = text.lower()
    engine_cc: Optional[float] = None
    km_driven: Optional[float] = None
    year: Optional[int] = None

    # 1. Extracción de Cilindraje (engine_cc)
    m_cc = re.search(r"\b([6-9]\d{2}|[1-5]\d{3})\s*(?:cc|cm3)\b", text_lower)
    if m_cc:
        engine_cc = float(m_cc.group(1))
    else:
        m_liters = re.search(
            r"(?:motor|cilindraje|cilindrada)?\s*\b([0-5]\.[0-9])\s*(?:l|litros?|cc)?\b",
            text_lower,
        )
        if m_liters and m_liters.group(1):
            engine_cc = round(float(m_liters.group(1)) * 1000.0, 1)
        else:
            m_motor = re.search(
                r"(?:motor|cilindraje|cilindrada)\s*(?:de)?\s*([6-9]\d{2}|[1-5]\d{3})\b",
                text_lower,
            )
            if m_motor:
                engine_cc = float(m_motor.group(1))

    # 2. Extracción de Kilometraje (km_driven)
    m_mil = re.search(
        r"\b(\d+(?:[\.,]\d+)?)\s*(?:mil|k)\s*(?:km|kms|kilometros?)?\b",
        text_lower,
    )
    if m_mil:
        val = float(m_mil.group(1).replace(",", "."))
        km_driven = val * 1000.0
    else:
        m_km = re.search(
            r"\b(\d{1,3}(?:\.\d{3})+|\d{4,7})\s*(?:km|kms|kilometros?)\b",
            text_lower,
        )
        if m_km:
            km_driven = float(m_km.group(1).replace(".", "").replace(",", ""))
        else:
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
        if engine_cc and abs(cand_val - engine_cc) < 1e-3 and "cc" in text_lower:
            continue
        year = cand_val
        break

    return VehicleParameters(year=year, km_driven=km_driven, engine_cc=engine_cc)


def _call_gemini_with_timeout(
    client: Any, prompt: str, timeout_sec: float = 3.0
) -> Optional[dict]:
    """Ejecuta la llamada a Gemini con un timeout rígido en hilo secundario."""
    def _worker():
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VehicleExtractionSchema,
            temperature=0.1,
        )
        # Invocar directamente el modelo activo sin pasar por modelos retirados
        res = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=config,
        )
        if res and res.text:
            return json.loads(res.text)
        return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_worker)
            return future.result(timeout=timeout_sec)
    except Exception:
        return None


def extract_parameters_llm(
    text: str, client: Any, timeout_sec: float = 3.0
) -> Tuple[Optional[VehicleParameters], Optional[str]]:
    """Extrae parámetros usando Gemini 3.8 Flash con límite de tiempo estricto."""
    if client is None:
        return None, None

    prompt = (
        "Eres un analista automotriz. Extrae los siguientes datos del vehículo:\n"
        "- year: Año (entero entre 1995 y 2026, o null)\n"
        "- km_driven: Kilometraje en km (número, ej. 60000 para '60 mil km', o null)\n"
        "- engine_cc: Cilindraje en cc (número, ej. 1600 para 1.6L o 1600 cc, o null)\n\n"
        f"Texto: \"{text}\""
    )

    data = _call_gemini_with_timeout(client, prompt, timeout_sec=timeout_sec)
    if data:
        params = VehicleParameters(
            year=data.get("year"),
            km_driven=float(data["km_driven"]) if data.get("km_driven") is not None else None,
            engine_cc=float(data["engine_cc"]) if data.get("engine_cc") is not None else None,
        )
        return params, "LLM (gemini-3.8-flash)"

    return None, None


def extract_parameters(
    user_message: str,
    client: Optional[Any] = None,
    mode: str = "hybrid",
) -> Tuple[VehicleParameters, str, float]:
    """
    Estrategia de extracción de alto rendimiento:
    - 'fast': Regex directo en ~2 ms.
    - 'hybrid' (Recomendado): Fast-Path Regex primero. Si detecta parámetros, responde de inmediato (<5 ms).
      Solo consulta a Gemini si el mensaje es ambiguo o no detectó parámetros, con timeout de 3s.
    - 'llm': Intenta primero con Gemini con timeout estricto de 3s; ante demora o fallo, recurre a Regex.
    """
    t_start = time.perf_counter()

    # MODO 1: Rápido (Regex Puro)
    if mode == "fast" or client is None:
        params = extract_parameters_regex(user_message)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return params, "Local (Regex)", elapsed_ms

    # MODO 2: Híbrido Inteligente (Fast-Path First)
    if mode == "hybrid":
        regex_params = extract_parameters_regex(user_message)
        # Si el regex extrajo información útil, responde inmediatamente sin esperar llamada de red
        if regex_params.has_any():
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return regex_params, "Fast-Path Local", elapsed_ms

        # Si el regex no encontró nada, consultar a Gemini con timeout estricto
        llm_params, method = extract_parameters_llm(user_message, client, timeout_sec=2.5)
        if llm_params is not None and llm_params.has_any():
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return llm_params, method, elapsed_ms

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return regex_params, "Local (Regex)", elapsed_ms

    # MODO 3: LLM Prioritario con Timeout Estricto
    llm_params, method = extract_parameters_llm(user_message, client, timeout_sec=3.0)
    if llm_params is not None:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return llm_params, method, elapsed_ms

    params = extract_parameters_regex(user_message)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return params, "Local (Timeout Fallback)", elapsed_ms


def answer_general_question_llm(
    text: str, client: Any, timeout_sec: float = 3.0
) -> Optional[str]:
    """Responde consultas conversacionales generales de vehículos usando Gemini con timeout."""
    if client is None:
        return None

    def _worker():
        prompt = (
            "Eres un asesor automotriz experto y amigable en Colombia. "
            "Responde de forma muy concisa (máximo 3 párrafos cortos) la siguiente duda del usuario sobre vehículos:\n\n"
            f"Usuario: \"{text}\""
        )
        res = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        return res.text if res else None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_worker)
            return future.result(timeout=timeout_sec)
    except Exception:
        return None


def build_missing_info_response(missing: List[str], elapsed_ms: float, method: str) -> str:
    """Genera una respuesta cordial solicitando los datos faltantes."""
    fields_list = "\n".join([f"- **{field}**" for field in missing])
    return (
        "Para realizar el cálculo con precisión, requiero los siguientes datos adicionales:\n\n"
        f"{fields_list}\n\n"
        "Por favor indícame estos valores para generar el informe comparativo.\n\n"
        f"> *Tiempo de respuesta: {elapsed_ms:.1f} ms | Motor: {method}*"
    )


def build_valuation_response(
    year: int,
    km_driven: float,
    engine_cc: float,
    estimated_price: float,
    similar_df: pd.DataFrame,
    extraction_method: str,
    elapsed_ms: float,
) -> str:
    """Redacta una respuesta analítica y estructurada con el avalúo y la comparativa KNN."""
    avg_similar_price = float(similar_df["price_cop"].mean())
    min_similar_price = float(similar_df["price_cop"].min())
    max_similar_price = float(similar_df["price_cop"].max())
    diff_cop = estimated_price - avg_similar_price
    pct_diff = ((estimated_price - avg_similar_price) / avg_similar_price) * 100.0

    if pct_diff < -3.0:
        recomendacion = (
            f"**Precio Competitivo:** El valor estimado se ubica un "
            f"**{abs(pct_diff):.1f}% por debajo** del promedio de vehículos similares en el mercado "
            f"(${avg_similar_price:,.0f} COP).\n"
            "- *Compradores:* Representa un precio favorable frente a la oferta actual.\n"
            "- *Vendedores:* Si el auto está en óptimo estado mecánico y estético, podrías fijar un precio "
            f"cercano a los ${avg_similar_price:,.0f} COP."
        )
    elif pct_diff > 3.0:
        recomendacion = (
            f"**Precio Superior al Promedio:** El valor estimado se sitúa un **{pct_diff:.1f}% por encima** "
            f"del promedio de los modelos comparables (${avg_similar_price:,.0f} COP).\n"
            "- *Compradores:* Conviene verificar historial de mantenimiento, accesorios y estado general.\n"
            "- *Vendedores:* El modelo tiene buena valorización; destaca su bajo desgaste al promocionarlo."
        )
    else:
        recomendacion = (
            f"**Precio en Equilibrio:** El avalúo estimado (${estimated_price:,.0f} COP) está alineado "
            f"con el promedio de modelos similares (${avg_similar_price:,.0f} COP), indicando un valor de "
            "mercado balanceado para una negociación rápida."
        )

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
        f"### Resultado del Avalúo Comercial\n\n"
        f"- **Año:** `{year}`\n"
        "- **Kilometraje:** `" f"{km_driven:,.0f} km`\n"
        "- **Cilindraje:** `" f"{engine_cc:,.0f} cc`\n\n"
        f"### Precio Estimado: **${estimated_price:,.0f} COP**\n\n"
        f"---\n\n"
        f"### Vehículos Más Similares en el Mercado (KNN)\n"
        f"Rango de mercado de similares: **${min_similar_price:,.0f}** - **${max_similar_price:,.0f} COP** "
        f"(Promedio: **${avg_similar_price:,.0f} COP**).\n\n"
        f"{tabla_md}\n\n"
        f"---\n\n"
        f"### Recomendación de Mercado\n"
        f"{recomendacion}\n\n"
        f"> *Tiempo de respuesta: **{elapsed_ms:.1f} ms** | Motor: **{extraction_method}***"
    )

    return response


def process_user_message(
    user_message: str,
    models_bundle: Dict[str, Any],
    df: pd.DataFrame,
    current_params: Optional[VehicleParameters] = None,
    client: Optional[Any] = None,
    mode: str = "hybrid",
) -> Tuple[str, VehicleParameters, Optional[pd.DataFrame]]:
    """
    Función orquestadora del chatbot con optimización de latencia:
    1. Extracción con Fast-Path First (Modo Híbrido ultrarrápido).
    2. Si es una pregunta conversacional sin datos vehiculares y hay cliente LLM, responde con Gemini.
    3. Acumulación y resolución multi-turno instantánea.
    4. Avalúo ML con Regresión Lineal + KNN en milisegundos.
    """
    from src.models import predict_and_find_similar

    # 1. Extraer del mensaje con la estrategia seleccionada
    new_params, method, elapsed_ms = extract_parameters(
        user_message=user_message, client=client, mode=mode
    )

    # 2. Si no se detectó ningún dato vehicular y tampoco hay datos previos:
    if not new_params.has_any() and (current_params is None or not current_params.has_any()):
        # Si es una consulta conversacional y hay cliente LLM, responder con Gemini
        if client is not None and mode in ("hybrid", "llm"):
            general_answer = answer_general_question_llm(user_message, client, timeout_sec=3.0)
            if general_answer:
                response = (
                    f"{general_answer}\n\n"
                    "---\n"
                    "💡 *Si deseas tasar un vehículo específico, puedes indicarme su **año**, **kilometraje** y **cilindraje**.*"
                )
                return response, VehicleParameters(), None

    # 3. Fusionar con parámetros acumulados de la conversación
    if current_params is not None:
        accumulated_params = current_params.merge(new_params)
    else:
        accumulated_params = new_params

    # 4. Comprobar si faltan parámetros
    if not accumulated_params.is_complete():
        missing = accumulated_params.missing_fields()
        response_text = build_missing_info_response(missing, elapsed_ms, method)
        return response_text, accumulated_params, None

    # 5. Si están completos, llamar a los modelos de ML
    t_ml_start = time.perf_counter()
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
    total_elapsed_ms = elapsed_ms + (time.perf_counter() - t_ml_start) * 1000.0

    response_text = build_valuation_response(
        year=year,
        km_driven=km,
        engine_cc=cc,
        estimated_price=pred_price,
        similar_df=similar_df,
        extraction_method=method,
        elapsed_ms=total_elapsed_ms,
    )

    return response_text, VehicleParameters(), similar_df


def main():
    """Prueba de latencia y funcionamiento del módulo optimizado."""
    from src.models import train_and_evaluate

    data_path = PROJECT_ROOT / "data" / "processed" / "vehicles_unified.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

    print("Cargando dataset y modelos...")
    df = pd.read_csv(data_path)
    bundle = train_and_evaluate(df)
    client = get_gemini_client()

    print(f"Cliente Gemini disponible: {client is not None}")
    print("=" * 60)
    print("TEST DE RENDIMIENTO (FAST-PATH VS LLM)")
    print("=" * 60)

    test_queries = [
        "Cuánto vale un carro 2018 con 60 mil km y motor 1.6?",
        "Tengo un 2016 con 45k km y 2000 cc",
        "Auto 2020 con 15.000 km y motor 1.4 litros",
    ]

    for q in test_queries:
        t0 = time.perf_counter()
        resp, _, _ = process_user_message(q, bundle, df, client=client, mode="hybrid")
        total_time = (time.perf_counter() - t0) * 1000.0
        print(f"Query: \"{q}\"")
        print(f"Tiempo Total de Respuesta: {total_time:.2f} ms")
        print("-" * 60)


if __name__ == "__main__":
    main()
