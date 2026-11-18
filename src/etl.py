"""
Módulo ETL (Extract, Transform, Load) para el Tasador y Buscador de Vehículos Usados.

Procesa y unifica los datasets de:
1. audi.csv (Reino Unido - GBP a COP, millas a km, litros a cc)
2. Car details v3.csv (India - INR a COP, extracción de CC)
3. CAR DETAILS FROM CAR DEKHO.csv (India - INR a COP, extracción regex de CC)

Genera el dataset limpio y consolidado en data/processed/vehicles_unified.csv.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

# Configuración de Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "vehicles_unified.csv"

# Constantes de Conversión de Monedas y Unidades
GBP_TO_COP = 5400.0
INR_TO_COP = 50.0
MILES_TO_KM = 1.60934
MIN_PRICE_COP_THRESHOLD = 5_000_000.0

STANDARD_COLUMNS = [
    "name",
    "year",
    "price_cop",
    "km_driven",
    "engine_cc",
    "fuel",
    "transmission",
    "source",
]


def load_and_clean_audi(file_path: Path) -> pd.DataFrame:
    """Procesa el dataset de vehículos Audi (audi.csv)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    df = pd.read_csv(file_path)

    # 1. Limpieza de modelo y creación de name
    name = "Audi " + df["model"].astype(str).str.strip()

    # 2. Conversión de millas a km
    km_driven = (df["mileage"] * MILES_TO_KM).round(2)

    # 3. Conversión de motor de litros a cc
    engine_cc = (df["engineSize"] * 1000.0).round(1)

    # 4. Conversión de precio de GBP a COP
    price_cop = (df["price"] * GBP_TO_COP).round(2)

    processed_df = pd.DataFrame(
        {
            "name": name,
            "year": df["year"].astype(int),
            "price_cop": price_cop,
            "km_driven": km_driven,
            "engine_cc": engine_cc,
            "fuel": df["fuelType"],
            "transmission": df["transmission"],
            "source": "audi.csv",
        }
    )

    # Filtrar registros donde engine_cc <= 0
    processed_df = processed_df[processed_df["engine_cc"] > 0]
    return processed_df


def load_and_clean_car_details_v3(file_path: Path) -> pd.DataFrame:
    """Procesa el dataset de Car details v3.csv."""
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    df = pd.read_csv(file_path)

    # 1. Extracción de valor numérico de engine eliminando ' CC'
    engine_cc = (
        df["engine"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

    # 2. Conversión de INR a COP
    price_cop = (df["selling_price"] * INR_TO_COP).round(2)

    processed_df = pd.DataFrame(
        {
            "name": df["name"].astype(str).str.strip(),
            "year": df["year"].astype(int),
            "price_cop": price_cop,
            "km_driven": df["km_driven"].astype(float).round(2),
            "engine_cc": engine_cc,
            "fuel": df["fuel"],
            "transmission": df["transmission"],
            "source": "Car details v3.csv",
        }
    )

    # Eliminar filas con nulos en engine_cc
    processed_df = processed_df.dropna(subset=["engine_cc"])
    return processed_df


def load_and_clean_car_dekho(file_path: Path) -> pd.DataFrame:
    """Procesa el dataset CAR DETAILS FROM CAR DEKHO.csv."""
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    df = pd.read_csv(file_path)

    # 1. Extracción de cilindraje con expresión regular en name (ej. 1.6 -> 1600, 1.2 -> 1200)
    # Busca patrones decimales correspondientes a litros (ej: 1.0, 1.2, 1.6, 2.0)
    liters_extracted = (
        df["name"]
        .astype(str)
        .str.extract(r"\b([0-9]+\.[0-9]+)\b")[0]
        .astype(float)
    )
    engine_cc = (liters_extracted * 1000.0).round(1)

    # 2. Conversión de INR a COP
    price_cop = (df["selling_price"] * INR_TO_COP).round(2)

    processed_df = pd.DataFrame(
        {
            "name": df["name"].astype(str).str.strip(),
            "year": df["year"].astype(int),
            "price_cop": price_cop,
            "km_driven": df["km_driven"].astype(float).round(2),
            "engine_cc": engine_cc,
            "fuel": df["fuel"],
            "transmission": df["transmission"],
            "source": "CAR DETAILS FROM CAR DEKHO.csv",
        }
    )

    # Descartar filas donde no sea posible determinar un engine_cc real
    processed_df = processed_df.dropna(subset=["engine_cc"])
    processed_df = processed_df[processed_df["engine_cc"] > 0]
    return processed_df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza categorías de fuel y transmission.
    - fuel: Petrol -> Gasolina, Diesel -> Diésel, Hybrid -> Híbrido, otros -> Otro
    - transmission: Manual -> Manual, Automatic -> Automática, Semi-Auto -> Semi-Automática
    """
    df = df.copy()

    fuel_mapping = {
        "Petrol": "Gasolina",
        "Diesel": "Diésel",
        "Hybrid": "Híbrido",
    }
    df["fuel"] = df["fuel"].astype(str).str.strip().map(lambda x: fuel_mapping.get(x, "Otro"))

    trans_mapping = {
        "Manual": "Manual",
        "Automatic": "Automática",
        "Semi-Auto": "Semi-Automática",
    }
    df["transmission"] = (
        df["transmission"]
        .astype(str)
        .str.strip()
        .map(lambda x: trans_mapping.get(x, x))
    )

    return df


def remove_duplicates_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados exactos y outliers extremos:
    - price_cop <= 5,000,000
    - km_driven < 0
    """
    df = df.copy()

    # Eliminar duplicados exactos en las variables descriptivas
    feature_columns = [
        "name",
        "year",
        "price_cop",
        "km_driven",
        "engine_cc",
        "fuel",
        "transmission",
    ]
    df = df.drop_duplicates(subset=feature_columns)

    # Filtrar outliers
    valid_mask = (df["price_cop"] > MIN_PRICE_COP_THRESHOLD) & (df["km_driven"] >= 0)
    df = df[valid_mask]

    return df


def run_etl_pipeline(raw_dir: Path = RAW_DATA_DIR, output_path: Path = OUTPUT_FILE) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ETL y guarda el dataset unificado.
    """
    print("=" * 60)
    print("INICIANDO PIPELINE ETL DE VEHÍCULOS USADOS")
    print("=" * 60)

    # 1. Cargar y procesar cada dataset
    print(f"[1/4] Procesando audi.csv...")
    df_audi = load_and_clean_audi(raw_dir / "audi.csv")
    print(f"      -> Registros procesados (Audi): {len(df_audi):,}")

    print(f"[2/4] Procesando Car details v3.csv...")
    df_v3 = load_and_clean_car_details_v3(raw_dir / "Car details v3.csv")
    print(f"      -> Registros procesados (Car details v3): {len(df_v3):,}")

    print(f"[3/4] Procesando CAR DETAILS FROM CAR DEKHO.csv...")
    df_dekho = load_and_clean_car_dekho(raw_dir / "CAR DETAILS FROM CAR DEKHO.csv")
    print(f"      -> Registros procesados (Car Dekho con CC real): {len(df_dekho):,}")

    # 2. Unificar datasets
    unified = pd.concat([df_audi, df_v3, df_dekho], ignore_index=True)
    total_raw_rows = len(unified)
    print(f"\n[4/4] Consolidando datos...")
    print(f"      -> Total antes de estandarizar y filtrar: {total_raw_rows:,}")

    # 3. Estandarizar categorías
    unified = standardize_categories(unified)

    # 4. Eliminar duplicados y outliers
    unified = remove_duplicates_and_outliers(unified)
    total_clean_rows = len(unified)
    print(f"      -> Total registros limpios y consolidados: {total_clean_rows:,}")
    print(f"      -> Filas eliminadas (duplicados / outliers): {total_raw_rows - total_clean_rows:,}")

    # 5. Asegurar orden de columnas estándar
    unified = unified[STANDARD_COLUMNS]

    # 6. Guardar dataset en disco
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(output_path, index=False)
    print(f"\nDataset unificado guardado exitosamente en:\n{output_path.resolve()}")

    # 7. Estadísticas finales
    print_dataset_statistics(unified)

    return unified


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """Imprime estadísticas detalladas del dataset unificado."""
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS FINALES DEL DATASET UNIFICADO")
    print("=" * 60)

    print("\n1. TOTAL DE FILAS POR FUENTE:")
    source_counts = df["source"].value_counts()
    for src, count in source_counts.items():
        pct = (count / len(df)) * 100
        print(f"   - {src:<32}: {count:>6,} filas ({pct:>5.1f}%)")
    print(f"   -> TOTAL: {len(df):,} filas")

    print("\n2. VALORES NULOS POR COLUMNA:")
    null_counts = df.isna().sum()
    for col, nulls in null_counts.items():
        print(f"   - {col:<15}: {nulls:>5} nulos")

    print("\n3. RANGOS MIN / MAX Y PROMEDIOS DE VARIABLES NUMÉRICAS:")
    num_cols = ["year", "price_cop", "km_driven", "engine_cc"]
    stats_df = df[num_cols].agg(["min", "mean", "median", "max"]).T
    stats_df["min"] = stats_df["min"].apply(lambda x: f"{x:,.2f}")
    stats_df["mean"] = stats_df["mean"].apply(lambda x: f"{x:,.2f}")
    stats_df["median"] = stats_df["median"].apply(lambda x: f"{x:,.2f}")
    stats_df["max"] = stats_df["max"].apply(lambda x: f"{x:,.2f}")
    print(stats_df.to_string())

    print("\n4. DISTRIBUCIÓN POR TIPO DE COMBUSTIBLE (fuel):")
    for fuel, count in df["fuel"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"   - {fuel:<15}: {count:>6,} ({pct:>5.1f}%)")

    print("\n5. DISTRIBUCIÓN POR TIPO DE TRANSMISIÓN (transmission):")
    for trans, count in df["transmission"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"   - {trans:<20}: {count:>6,} ({pct:>5.1f}%)")

    print("\n6. PRIMERAS 5 FILAS DEL DATASET UNIFICADO:")
    print(df.head(5).to_string())
    print("=" * 60)


if __name__ == "__main__":
    run_etl_pipeline()
