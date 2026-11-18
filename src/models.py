"""
Módulo de Machine Learning para el Tasador y Buscador de Vehículos Usados.

Contiene:
1. Modelo de Regresión Lineal para estimación de precios en COP.
2. Modelo K-Nearest Neighbors (KNN) con normalización (StandardScaler) para recomendación
   y búsqueda de los 5 vehículos más similares del mercado.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Constantes del módulo
FEATURE_COLS = ["year", "km_driven", "engine_cc"]
TARGET_COL = "price_cop"
DEFAULT_MIN_PRICE_FLOOR = 5_000_000.0  # Piso mínimo razonable en COP


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Entrena el modelo de Regresión Lineal y el modelo KNN para búsqueda de similares.

    Args:
        df: DataFrame con los datos limpios y unificados de vehículos.

    Returns:
        dict: Diccionario que contiene:
            - 'regression_model': Modelo LinearRegression entrenado.
            - 'knn_model': Modelo NearestNeighbors entrenado.
            - 'scaler': StandardScaler ajustado sobre las variables predictoras.
            - 'metrics': Métricas de evaluación (R2, MAE, RMSE, coeficientes, intercepto).
            - 'test_data': Diccionario con X_test, y_test y y_pred para graficar.
            - 'feature_cols': Lista de variables predictoras utilizadas.
    """
    # 1. Preparación de variables
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # 2. División train/test (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 3. Entrenamiento de Regresión Lineal
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # 4. Predicciones y métricas sobre el conjunto de prueba
    y_pred = reg_model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Coeficientes asociados a cada variable
    coef_dict = {
        col: float(coef) for col, coef in zip(FEATURE_COLS, reg_model.coef_)
    }
    intercept = float(reg_model.intercept_)

    # 5. Normalización y entrenamiento de KNN para recomendación
    # Se ajusta el escalador sobre el catálogo de vehículos disponibles
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn_model.fit(X_scaled)

    # 6. Empaquetado de resultados
    bundle = {
        "regression_model": reg_model,
        "knn_model": knn_model,
        "scaler": scaler,
        "metrics": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "coefficients": coef_dict,
            "intercept": intercept,
        },
        "test_data": {
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        },
        "feature_cols": FEATURE_COLS,
    }

    return bundle


def predict_and_find_similar(
    year: int,
    km_driven: float,
    engine_cc: float,
    models_bundle: Dict[str, Any],
    df: pd.DataFrame,
    min_floor: float = DEFAULT_MIN_PRICE_FLOOR,
) -> Tuple[float, pd.DataFrame]:
    """
    Estima el valor comercial del vehículo y encuentra los 5 más similares en el mercado.

    Args:
        year: Año de fabricación del vehículo.
        km_driven: Kilometraje recorrido en kilómetros.
        engine_cc: Cilindraje del motor en centímetros cúbicos (cc).
        models_bundle: Diccionario generado por train_and_evaluate.
        df: DataFrame del catálogo de vehículos.
        min_floor: Valor mínimo razonable en COP para la estimación.

    Returns:
        tuple: (precio_estimado_cop, df_similares)
            - precio_estimado_cop: float con la predicción acotada al piso mínimo.
            - df_similares: DataFrame con los 5 vehículos más cercanos y sus métricas de similitud.
    """
    reg_model: LinearRegression = models_bundle["regression_model"]
    knn_model: NearestNeighbors = models_bundle["knn_model"]
    scaler: StandardScaler = models_bundle["scaler"]

    # 1. Predicción de precio con Regresión Lineal
    input_features = pd.DataFrame(
        [[year, km_driven, engine_cc]], columns=FEATURE_COLS
    )
    raw_prediction = float(reg_model.predict(input_features)[0])
    estimated_price = max(raw_prediction, min_floor)

    # 2. Búsqueda de vehículos similares con KNN
    input_scaled = scaler.transform(input_features)
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=5)

    # 3. Construcción del DataFrame de vehículos similares
    similar_indices = indices[0]
    similar_distances = distances[0]

    output_cols = [
        "name",
        "year",
        "km_driven",
        "engine_cc",
        "fuel",
        "transmission",
        "price_cop",
    ]

    # Extraer los registros correspondientes del catálogo
    similar_df = df.iloc[similar_indices][output_cols].copy().reset_index(drop=True)

    # Agregar distancia euclidiana y porcentaje de similitud normalizado
    # Similitud: 100 / (1 + distancia) -> Distancia 0 = 100%, Distancia 1 = 50%
    similar_df["distance"] = np.round(similar_distances, 3)
    similar_df["similarity_pct"] = np.round(100.0 / (1.0 + similar_distances), 1)

    return estimated_price, similar_df


def main():
    """Ejecución de demostración y prueba del módulo."""
    data_path = PROJECT_ROOT / "data" / "processed" / "vehicles_unified.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {data_path}. Ejecuta primero src/etl.py."
        )

    print("Cargando dataset para entrenamiento...")
    df = pd.read_csv(data_path)
    print(f"Dataset cargado con {len(df):,} filas.")

    print("\nEntrenando modelos de Regresión Lineal y KNN...")
    bundle = train_and_evaluate(df)
    metrics = bundle["metrics"]

    print("\n" + "=" * 60)
    print("MÉTRICAS DE REGRESIÓN LINEAL:")
    print("=" * 60)
    print(f"R² Score              : {metrics['r2']:.4f}")
    print(f"MAE (Error Abs. Medio): ${metrics['mae']:,.2f} COP")
    print(f"RMSE (Error Cuadrático): ${metrics['rmse']:,.2f} COP")
    print(f"Intercepto (β₀)       : {metrics['intercept']:,.2f}")
    print("\nCoeficientes por Variable Predictora:")
    for col, coef in metrics["coefficients"].items():
        print(f"  - {col:<12}: {coef:>15,.2f} COP por unidad")

    # Prueba de inferencia
    test_year = 2018
    test_km = 60000.0
    test_cc = 1600.0

    print("\n" + "=" * 60)
    print(f"PREDICCIÓN DE EJEMPLO: Año {test_year}, {test_km:,.0f} km, {test_cc:,.0f} cc")
    print("=" * 60)

    pred_price, similar_cars = predict_and_find_similar(
        year=test_year,
        km_driven=test_km,
        engine_cc=test_cc,
        models_bundle=bundle,
        df=df,
    )

    print(f"Precio Estimado: ${pred_price:,.2f} COP\n")
    print("Top 5 Vehículos Más Similares del Mercado:")
    print(similar_cars.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
