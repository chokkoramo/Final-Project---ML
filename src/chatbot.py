"""
Módulo del Agente Autónomo con Function Calling para el Tasador de Vehículos Usados.

Implementa un Agente de IA basado en herramientas (Tool-Using Agent) usando el SDK
`google-genai` con el modelo `gemini-3.8-flash` (reemplazo activo indicado por la API
para `gemini-2.5-flash`, que devuelve 404 NOT_FOUND). El modelo razona sobre la
intención del usuario, solicita datos faltantes cuando corresponde y decide autónomamente
qué herramientas invocar y en qué orden:

1. `tool_estimar_precio_lineal`: Avalúo comercial en COP y MAE (Regresión Lineal).
2. `tool_buscar_vehiculos_similares`: Búsqueda de los `k` vehículos más cercanos (KNN).
3. `tool_consultar_estadisticas_mercado`: Estadísticas agregadas del catálogo unificado.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from src.models import (
    DEFAULT_MIN_PRICE_FLOOR,
    FEATURE_COLS,
    train_and_evaluate,
)

# Carga de variables de entorno desde .env
load_dotenv()

# Importación segura del SDK google-genai
try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    GENAI_AVAILABLE = False

# Constantes del Agente
# Nota: La API de Google GenAI retiró `gemini-2.5-flash` para nuevos usuarios (404 NOT_FOUND)
# indicando migrar a `gemini-3.8-flash`. Se permite sobreescribir el modelo en `create_agent_chat`.
AGENT_MODEL = "gemini-3.8-flash"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "vehicles_unified.csv"
MAX_SIMILAR_RESULTS = 20
MIN_VALID_YEAR = 1980
MAX_VALID_YEAR = 2026

MISSING_API_KEY_MESSAGE = (
    "⚠️ **Modo agéntico no disponible:** No se encontró una clave `GEMINI_API_KEY` válida.\n\n"
    "Para utilizar el asesor autónomo con Google Gemini y Function Calling, configura la variable "
    "`GEMINI_API_KEY` en el archivo `.env` del proyecto o ingrésala en la **barra lateral**."
)

SYSTEM_INSTRUCTION = """
Eres un asesor experto en valoración y análisis del mercado de vehículos usados en Colombia.
Respondes siempre en español, de manera clara, profesional, amable y basada estrictamente en los datos devueltos por tus herramientas.

Herramientas disponibles:
1. `tool_estimar_precio_lineal(year, km_driven, engine_cc)`: Calcula el avalúo comercial estimado en pesos colombianos (COP) mediante Regresión Lineal y entrega el margen de error (MAE) y métricas del modelo.
2. `tool_buscar_vehiculos_similares(year, km_driven, engine_cc, k=5)`: Busca en el catálogo unificado los `k` vehículos reales más parecidos usando KNN (`NearestNeighbors`).
3. `tool_consultar_estadisticas_mercado(marca=None, year=None)`: Consulta estadísticas reales (cantidad, precio promedio, mediana, mínimo y máximo) en el dataset `vehicles_unified.csv` filtrando opcionalmente por marca/modelo y/o año.

Reglas de comportamiento obligatorio:
1. **Validación de datos antes de llamar herramientas:**
   - Para usar `tool_estimar_precio_lineal` o `tool_buscar_vehiculos_similares`, necesitas obligatoriamente **año (`year`)**, **kilometraje (`km_driven`)** y **cilindraje (`engine_cc`)**.
   - Si falta alguno de estos tres datos (y no fue proporcionado en turnos anteriores de la conversación), NO inventes ni asumas valores: pídele amablemente al usuario los datos faltantes antes de invocar dichas herramientas.
   - Para `tool_consultar_estadisticas_mercado`, NO necesitas kilometraje ni cilindraje; `marca` y `year` son opcionales. Si el usuario pregunta por el comportamiento general del mercado o por una marca/año específico (por ejemplo, *"¿Cómo están los precios de los Audi 2018 en el mercado?"*), llama directamente a `tool_consultar_estadisticas_mercado`.

2. **Normalización de unidades antes de invocar herramientas:**
   - Convierte expresiones coloquiales del usuario a unidades numéricas exactas:
     - Kilometraje: `"60 mil km"` o `"60k"` -> `60000.0`.
     - Cilindraje: `"1.6"`, `"1.6L"`, `"1.6 litros"` -> `1600.0`; `"2.0"` -> `2000.0`.

3. **Razonamiento autónomo y encadenamiento:**
   - Decide según la petición si basta con una sola herramienta o si conviene encadenar varias:
     - Si piden una tasación completa, estima el precio con `tool_estimar_precio_lineal`, busca referentes comparables con `tool_buscar_vehiculos_similares` y compara el avalúo lineal frente al promedio de los similares o del mercado.
     - Si el usuario ya tasó un vehículo en un turno previo y pide algo como *"ahora muéstrame los 3 más parecidos a ese"*, reutiliza los parámetros de la conversación e invoca `tool_buscar_vehiculos_similares(..., k=3)`.
     - Si solo piden estadísticas del mercado, invoca únicamente `tool_consultar_estadisticas_mercado`.

4. **Transparencia y formato:**
   - Nunca inventes precios, estadísticas ni vehículos; reporta únicamente cifras provenientes de las herramientas.
   - Formatea los montos en pesos colombianos de forma legible (ej. `$68,500,000 COP`), menciona el margen de error (MAE) cuando presentes un avalúo lineal e indica si el valor estimado está por encima o por debajo del promedio de los vehículos comparables.
""".strip()

# Contexto de ejecución (datos, modelos y trazabilidad por turno)
_DEFAULT_CONTEXT: Dict[str, Any] = {
    "df": None,
    "models_bundle": None,
}
_runtime_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "chatbot_runtime_context", default=None
)
_turn_tool_trace: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "chatbot_turn_tool_trace", default=None
)


def configure_agent_context(
    models_bundle: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Configura en memoria el DataFrame unificado y el bundle de modelos ML
    para que las herramientas del agente los utilicen sin reentrenar en cada turno.
    """
    current = dict(_runtime_context.get() or _DEFAULT_CONTEXT)
    if df is not None:
        current["df"] = df
        _DEFAULT_CONTEXT["df"] = df
    if models_bundle is not None:
        current["models_bundle"] = models_bundle
        _DEFAULT_CONTEXT["models_bundle"] = models_bundle
    _runtime_context.set(current)


def _get_dataframe() -> pd.DataFrame:
    """Obtiene el DataFrame procesado desde el contexto o lo carga desde disco."""
    ctx = _runtime_context.get()
    if ctx and ctx.get("df") is not None:
        return ctx["df"]
    if _DEFAULT_CONTEXT.get("df") is not None:
        return _DEFAULT_CONTEXT["df"]

    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset unificado no encontrado en {DEFAULT_DATA_PATH}. "
            "Ejecuta primero `python src/etl.py`."
        )
    df = pd.read_csv(DEFAULT_DATA_PATH)
    _DEFAULT_CONTEXT["df"] = df
    return df


def _get_models_bundle() -> Dict[str, Any]:
    """Obtiene el bundle de modelos ML desde el contexto o los entrena bajo demanda."""
    ctx = _runtime_context.get()
    if ctx and ctx.get("models_bundle") is not None:
        return ctx["models_bundle"]
    if _DEFAULT_CONTEXT.get("models_bundle") is not None:
        return _DEFAULT_CONTEXT["models_bundle"]

    df = _get_dataframe()
    bundle = train_and_evaluate(df)
    _DEFAULT_CONTEXT["models_bundle"] = bundle
    return bundle


def _record_tool_usage(tool_name: str, args: Dict[str, Any], result: Any) -> None:
    """Registra la invocación de una herramienta en la traza del turno actual."""
    trace = _turn_tool_trace.get()
    if trace is not None:
        trace.append(
            {
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )


def _validate_vehicle_features(
    year: int, km_driven: float, engine_cc: float
) -> Optional[str]:
    """Valida rangos razonables para los parámetros numéricos de un vehículo."""
    if not (MIN_VALID_YEAR <= int(year) <= MAX_VALID_YEAR):
        return (
            f"El año {year} está fuera del rango válido ({MIN_VALID_YEAR}-{MAX_VALID_YEAR})."
        )
    if float(km_driven) < 0:
        return "El kilometraje no puede ser negativo."
    if float(engine_cc) <= 0:
        return "El cilindraje del motor debe ser mayor a 0 cc."
    return None


def tool_estimar_precio_lineal(
    year: int, km_driven: float, engine_cc: float
) -> dict:
    """
    Calcula el avalúo comercial estimado en pesos colombianos (COP) usando el modelo
    de Regresión Lineal (OLS) junto con su margen de error (MAE).

    Úsala cuando necesites estimar el valor comercial de un vehículo específico y
    conozcas sus tres características obligatorias. Si falta algún dato, pídelo al
    usuario antes de invocar esta herramienta.

    Args:
        year: Año de fabricación del vehículo (entero entre 1980 y 2026, ej. 2018).
        km_driven: Kilometraje total recorrido en kilómetros (ej. 60000.0 para '60 mil km').
        engine_cc: Cilindraje del motor en centímetros cúbicos (ej. 1600.0 para motor 1.6L).

    Returns:
        dict: Diccionario con `precio_estimado_cop`, `mae_cop` (margen de error absoluto medio),
        `rango_min_cop`, `rango_max_cop`, `r2_score` y los parámetros evaluados.
    """
    args = {
        "year": int(year),
        "km_driven": float(km_driven),
        "engine_cc": float(engine_cc),
    }
    validation_error = _validate_vehicle_features(
        args["year"], args["km_driven"], args["engine_cc"]
    )
    if validation_error:
        error_payload = {"error": validation_error, **args}
        _record_tool_usage("tool_estimar_precio_lineal", args, error_payload)
        return error_payload

    bundle = _get_models_bundle()
    reg_model = bundle["regression_model"]
    metrics = bundle["metrics"]

    input_features = pd.DataFrame(
        [[args["year"], args["km_driven"], args["engine_cc"]]],
        columns=FEATURE_COLS,
    )
    raw_prediction = float(reg_model.predict(input_features)[0])
    estimated_price = max(raw_prediction, DEFAULT_MIN_PRICE_FLOOR)
    mae = float(metrics["mae"])

    result = {
        "year": args["year"],
        "km_driven": args["km_driven"],
        "engine_cc": args["engine_cc"],
        "precio_estimado_cop": round(estimated_price, 2),
        "mae_cop": round(mae, 2),
        "rango_min_cop": round(max(DEFAULT_MIN_PRICE_FLOOR, estimated_price - mae), 2),
        "rango_max_cop": round(estimated_price + mae, 2),
        "r2_score": round(float(metrics["r2"]), 4),
        "modelo": "Regresión Lineal (OLS)",
    }
    _record_tool_usage("tool_estimar_precio_lineal", args, result)
    return result


def tool_buscar_vehiculos_similares(
    year: int, km_driven: float, engine_cc: float, k: int = 5
) -> list[dict]:
    """
    Encuentra los `k` vehículos reales más parecidos del catálogo unificado usando el
    modelo K-Nearest Neighbors (KNN con distancia euclidiana sobre variables estandarizadas).

    Úsala cuando el usuario pida ejemplos reales comparables, vehículos similares en el
    mercado o quiera contrastar un avalúo frente a autos reales. Requiere año,
    kilometraje y cilindraje completos.

    Args:
        year: Año de fabricación del vehículo de referencia (entero, ej. 2018).
        km_driven: Kilometraje recorrido en kilómetros (ej. 60000.0).
        engine_cc: Cilindraje del motor en cc (ej. 1600.0).
        k: Cantidad de vehículos similares a retornar (por defecto 5, máximo 20).

    Returns:
        list[dict]: Lista ordenada por cercanía con `name`, `year`, `km_driven`,
        `engine_cc`, `fuel`, `transmission`, `price_cop`, `distance` y `similarity_pct`.
    """
    requested_k = int(k)
    args = {
        "year": int(year),
        "km_driven": float(km_driven),
        "engine_cc": float(engine_cc),
        "k": requested_k,
    }
    validation_error = _validate_vehicle_features(
        args["year"], args["km_driven"], args["engine_cc"]
    )
    if validation_error:
        error_payload = [{"error": validation_error}]
        _record_tool_usage("tool_buscar_vehiculos_similares", args, error_payload)
        return error_payload
    if requested_k < 1:
        error_payload = [{"error": "El parámetro `k` debe ser mayor o igual a 1."}]
        _record_tool_usage("tool_buscar_vehiculos_similares", args, error_payload)
        return error_payload

    df = _get_dataframe()
    bundle = _get_models_bundle()
    knn_model = bundle["knn_model"]
    scaler = bundle["scaler"]

    safe_k = min(requested_k, MAX_SIMILAR_RESULTS, len(df))
    input_features = pd.DataFrame(
        [[args["year"], args["km_driven"], args["engine_cc"]]],
        columns=FEATURE_COLS,
    )
    input_scaled = scaler.transform(input_features)
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=safe_k)

    similar_records: list[dict] = []
    for idx, dist in zip(indices[0], distances[0]):
        row = df.iloc[int(idx)]
        dist_val = float(dist)
        similar_records.append(
            {
                "name": str(row["name"]),
                "year": int(row["year"]),
                "km_driven": round(float(row["km_driven"]), 2),
                "engine_cc": round(float(row["engine_cc"]), 1),
                "fuel": str(row["fuel"]),
                "transmission": str(row["transmission"]),
                "price_cop": round(float(row["price_cop"]), 2),
                "distance": round(dist_val, 3),
                "similarity_pct": round(float(100.0 / (1.0 + dist_val)), 1),
            }
        )

    _record_tool_usage("tool_buscar_vehiculos_similares", args, similar_records)
    return similar_records


def tool_consultar_estadisticas_mercado(
    marca: Optional[str] = None, year: Optional[int] = None
) -> dict:
    """
    Consulta el dataset procesado (`vehicles_unified.csv`) para calcular estadísticas
    reales del mercado (cantidad de vehículos, precio promedio, mediana, mínimo y máximo)
    aplicando filtros opcionales por marca/modelo y/o año.

    Úsala cuando el usuario pregunte por precios o tendencias generales del mercado,
    por una marca específica o por un año concreto (ej. "¿Cómo están los precios de
    los Audi 2018 en el mercado?" o "¿Cuál es el precio promedio en el catálogo?").

    Args:
        marca: Nombre de la marca o modelo a filtrar (insensible a mayúsculas, ej. "Audi",
            "Hyundai", "Audi A3"). Si es None, incluye todas las marcas.
        year: Año del modelo a filtrar (ej. 2018). Si es None, incluye todos los años.

    Returns:
        dict: Estadísticas de mercado (`cantidad_vehiculos`, `precio_promedio_cop`,
        `precio_mediana_cop`, `precio_minimo_cop`, `precio_maximo_cop`, etc.) y filtros aplicados.
    """
    marca_clean = marca.strip() if isinstance(marca, str) and marca.strip() else None
    year_clean = int(year) if year is not None else None
    args = {"marca": marca_clean, "year": year_clean}

    df = _get_dataframe()
    filtered = df

    if marca_clean is not None:
        filtered = filtered[
            filtered["name"].astype(str).str.contains(
                marca_clean, case=False, na=False, regex=False
            )
        ]
    if year_clean is not None:
        filtered = filtered[filtered["year"] == year_clean]

    if filtered.empty:
        empty_result = {
            "filtros": args,
            "cantidad_vehiculos": 0,
            "mensaje": (
                "No se encontraron vehículos en el dataset unificado con los filtros especificados."
            ),
        }
        _record_tool_usage("tool_consultar_estadisticas_mercado", args, empty_result)
        return empty_result

    sample_models = (
        filtered["name"].astype(str).value_counts().head(5).index.tolist()
    )
    result = {
        "filtros": args,
        "cantidad_vehiculos": int(len(filtered)),
        "precio_promedio_cop": round(float(filtered["price_cop"].mean()), 2),
        "precio_mediana_cop": round(float(filtered["price_cop"].median()), 2),
        "precio_minimo_cop": round(float(filtered["price_cop"].min()), 2),
        "precio_maximo_cop": round(float(filtered["price_cop"].max()), 2),
        "kilometraje_promedio": round(float(filtered["km_driven"].mean()), 2),
        "cilindraje_promedio_cc": round(float(filtered["engine_cc"].mean()), 1),
        "modelos_frecuentes": [str(m) for m in sample_models],
    }
    _record_tool_usage("tool_consultar_estadisticas_mercado", args, result)
    return result


AGENT_TOOLS = [
    tool_estimar_precio_lineal,
    tool_buscar_vehiculos_similares,
    tool_consultar_estadisticas_mercado,
]


def get_gemini_client(api_key_override: Optional[str] = None) -> Optional[Any]:
    """
    Inicializa y retorna un cliente de `google.genai` si `GEMINI_API_KEY` es válida.
    Retorna `None` si falta la librería o la clave de API.
    """
    if not GENAI_AVAILABLE:
        return None
    load_dotenv()
    api_key = api_key_override or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() in ("", "tu_api_key_aqui"):
        return None
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception:
        return None


def build_agent_config(system_instruction: Optional[str] = None) -> Any:
    """
    Construye la configuración `types.GenerateContentConfig` con las herramientas
    de ML y las instrucciones del asesor automotriz.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError(
            "El paquete `google-genai` no está instalado. Ejecuta `pip install google-genai`."
        )
    return types.GenerateContentConfig(
        tools=AGENT_TOOLS,
        system_instruction=system_instruction or SYSTEM_INSTRUCTION,
        temperature=0.2,
    )


def create_agent_chat(
    client: Optional[Any],
    model: str = AGENT_MODEL,
    system_instruction: Optional[str] = None,
) -> Optional[Any]:
    """
    Crea una sesión de chat con memoria conversacional usando `client.chats.create`.
    Por defecto utiliza `AGENT_MODEL` (`gemini-3.8-flash`).
    """
    if client is None or not GENAI_AVAILABLE:
        return None
    config = build_agent_config(system_instruction=system_instruction)
    return client.chats.create(model=model, config=config)


def _extract_response_text(response: Any) -> str:
    """Extrae de forma segura el texto final de una respuesta de Gemini."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts_text: List[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                parts_text.append(part_text.strip())
    if parts_text:
        return "\n\n".join(parts_text)

    return "No se recibió una respuesta textual del agente. Por favor intenta reformular tu consulta."


def _format_agent_error(exc: Exception) -> str:
    """Convierte errores del SDK o de red en un mensaje claro para el usuario."""
    msg = str(exc)
    lower = msg.lower()
    if any(tok in lower for tok in ("api_key", "api key", "permission_denied", "401", "403")):
        return (
            "⚠️ **Error de autenticación con Gemini:** Verifica que tu `GEMINI_API_KEY` "
            "en `.env` o en la barra lateral sea válida y tenga permisos activos."
        )
    if "429" in lower or "quota" in lower or "rate limit" in lower:
        return (
            "⚠️ **Límite de solicitudes alcanzado:** La cuota de la API de Gemini se agotó "
            "temporalmente. Intenta nuevamente en unos segundos."
        )
    if "503" in lower or "unavailable" in lower or "high demand" in lower:
        return (
            "⚠️ **Servicio temporalmente saturado (503):** El modelo de Gemini está "
            "experimentando alta demanda en este momento. Intenta nuevamente en unos segundos."
        )
    return (
        f"⚠️ **Error al ejecutar el agente:** {msg}\n\n"
        "Verifica tu conexión a internet y la configuración de `GEMINI_API_KEY`."
    )


def _is_transient_unavailable(exc: Exception) -> bool:
    """Detecta errores transitorios de disponibilidad (HTTP 503)."""
    lower = str(exc).lower()
    return "503" in lower or "unavailable" in lower or "high demand" in lower


def run_agent_turn(
    user_message: str,
    chat: Optional[Any] = None,
    client: Optional[Any] = None,
    models_bundle: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ejecuta un turno conversacional del Agente Autónomo con Function Calling.

    Args:
        user_message: Mensaje o pregunta en lenguaje natural del usuario.
        chat: Sesión de chat creada con `create_agent_chat` (mantiene memoria multi-turno).
        client: Cliente `genai.Client` opcional (usado para crear un chat si `chat` es None).
        models_bundle: Diccionario de modelos ML preentrenados (opcional).
        df: DataFrame del catálogo unificado (opcional).

    Returns:
        tuple: `(response_text, used_tools)`
            - `response_text`: Respuesta final redactada por el agente.
            - `used_tools`: Lista ordenada de herramientas invocadas en este turno,
              cada una con `{"tool": nombre, "args": argumentos, "result": salida}`.
    """
    if models_bundle is not None or df is not None:
        configure_agent_context(models_bundle=models_bundle, df=df)

    if not user_message or not user_message.strip():
        return "Por favor escribe una consulta sobre vehículos para poder ayudarte.", []

    if not GENAI_AVAILABLE:
        return (
            "⚠️ El paquete `google-genai` no está instalado en el entorno actual.",
            [],
        )

    active_chat = chat
    if active_chat is None:
        if client is None:
            return MISSING_API_KEY_MESSAGE, []
        active_chat = create_agent_chat(client)
        if active_chat is None:
            return MISSING_API_KEY_MESSAGE, []

    trace: List[Dict[str, Any]] = []
    token = _turn_tool_trace.set(trace)
    cleaned_message = user_message.strip()
    try:
        for attempt in range(2):
            try:
                response = active_chat.send_message(cleaned_message)
                response_text = _extract_response_text(response)
                return response_text, list(trace)
            except Exception as exc:
                # Reintentar una sola vez ante 503 solo si aún no se ejecutaron herramientas
                if attempt == 0 and not trace and _is_transient_unavailable(exc):
                    time.sleep(1.5)
                    continue
                return _format_agent_error(exc), list(trace)
        return _format_agent_error(RuntimeError("No se obtuvo respuesta del agente.")), list(trace)
    finally:
        _turn_tool_trace.reset(token)


@dataclass
class VehicleParameters:
    """
    Contenedor de compatibilidad para `app.py` previo a la Fase 5.
    Mantiene la sesión conversacional del agente (`chat_session`) entre turnos.
    """

    year: Optional[int] = None
    km_driven: Optional[float] = None
    engine_cc: Optional[float] = None
    chat_session: Optional[Any] = field(default=None, repr=False)
    _api_key_hash: Optional[int] = field(default=None, repr=False)


def process_user_message(
    user_message: str,
    models_bundle: Dict[str, Any],
    df: pd.DataFrame,
    current_params: Optional[VehicleParameters] = None,
    client: Optional[Any] = None,
    mode: str = "hybrid",
) -> Tuple[str, VehicleParameters, Optional[pd.DataFrame]]:
    """
    Adaptador de compatibilidad para la interfaz `app.py` actual (previo a Fase 5).
    Enruta las consultas hacia `run_agent_turn` manteniendo la memoria en `VehicleParameters`.
    """
    del mode  # El agente autónomo decide el flujo mediante Function Calling
    state = current_params if current_params is not None else VehicleParameters()

    if client is None:
        state.chat_session = None
        state._api_key_hash = None
        return MISSING_API_KEY_MESSAGE, state, None

    api_key = getattr(getattr(client, "_api_client", None), "api_key", None)
    key_hash = hash(api_key) if api_key is not None else None

    if state.chat_session is None or (key_hash is not None and state._api_key_hash != key_hash):
        state.chat_session = create_agent_chat(client)
        state._api_key_hash = key_hash

    response_text, _ = run_agent_turn(
        user_message=user_message,
        chat=state.chat_session,
        client=client,
        models_bundle=models_bundle,
        df=df,
    )
    return response_text, state, None


def main() -> None:
    """Prueba local de las herramientas y del agente (si existe GEMINI_API_KEY)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print("VERIFICACIÓN DE HERRAMIENTAS DEL AGENTE (FASE 4)")
    print("=" * 60)

    df = _get_dataframe()
    bundle = _get_models_bundle()
    configure_agent_context(models_bundle=bundle, df=df)

    print(f"Dataset cargado: {len(df):,} vehículos")
    print("\n1) Prueba tool_estimar_precio_lineal(2018, 60000, 1600):")
    print(tool_estimar_precio_lineal(2018, 60000.0, 1600.0))

    print("\n2) Prueba tool_buscar_vehiculos_similares(2018, 60000, 1600, k=3):")
    for item in tool_buscar_vehiculos_similares(2018, 60000.0, 1600.0, k=3):
        print("  -", item)

    print("\n3) Prueba tool_consultar_estadisticas_mercado(marca='Audi', year=2018):")
    print(tool_consultar_estadisticas_mercado(marca="Audi", year=2018))

    client = get_gemini_client()
    print(f"\nCliente Gemini disponible: {client is not None}")
    if client is None:
        resp, tools = run_agent_turn("Cuánto vale un carro 2018 con 60 mil km y motor 1.6?")
        print("Respuesta sin API Key:\n", resp)
        print("Herramientas usadas:", tools)
    else:
        chat = create_agent_chat(client)
        for q in [
            "¿Cómo están los precios de los Audi 2018 en el mercado?",
            "Tengo un carro 2018 con 60 mil km y motor 1.6, ¿cuánto vale?",
            "Ahora muéstrame los 3 más parecidos a ese.",
        ]:
            print(f"\nUsuario: {q}")
            resp, tools = run_agent_turn(q, chat=chat)
            print("Herramientas usadas:", [t["tool"] for t in tools])
            print("Agente:\n", resp)
    print("=" * 60)


if __name__ == "__main__":
    main()
