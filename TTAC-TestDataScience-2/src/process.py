"""Script para procesamiento de datos de series temporales."""

import argparse
from pathlib import Path

import numpy as np

from ttac_test_ds_timeseries.data import (
    clean_timeseries_data,
    load_air_quality_data,
    prepare_timeseries_features_air_quality,
    split_timeseries_data,
    validate_timeseries_data,
)


def process_air_quality_data():
    """Procesar datos de Air Quality UCI dataset."""
    print("📊 Iniciando procesamiento de Air Quality UCI...")

    # 1. Cargar datos
    df = load_air_quality_data(save_to_disk=True, target_column="CO(GT)")
    print(f"✅ Datos cargados: {df.shape}")

    # 2. Validar calidad de datos
    quality_metrics = validate_timeseries_data(df, date_column="datetime")
    print("📋 Métricas de calidad:")
    for key, value in quality_metrics.items():
        print(f"   {key}: {value}")

    # 3. Limpiar datos
    df_clean = clean_timeseries_data(
        df, date_column="datetime", fill_method="interpolate")
    print(f"🧹 Datos limpiados: {df_clean.shape}")

    # 4. Dividir en entrenamiento y prueba
    train_df, test_df = split_timeseries_data(
        df_clean, train_ratio=0.8, date_column="datetime")

    # 5. Preparar características para modelos
    try:
        X, y = prepare_timeseries_features_air_quality(
            df_clean, target_column="CO(GT)", sequence_length=100)
        print(f"🔧 Características preparadas: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"❌ Error preparando características: {e}")

    return df_clean, train_df, test_df


def main():
    """Procesar datos de series temporales."""
    parser = argparse.ArgumentParser(
        description="Procesar datos de series temporales")
    parser.add_argument(
        "--source",
        type=str,
        default="air_quality",
        choices=["air_quality"],
        help="Fuente de datos a procesar")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Directorio de salida para datos procesados")

    args = parser.parse_args()

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Procesar según la fuente
    if args.source == "air_quality":
        df_clean, train_df, test_df = process_air_quality_data()
        print("✅ Procesamiento de Air Quality completado")

        # Guardar datos procesados
        clean_file = output_dir / "air_quality_cleaned.csv"
        train_file = output_dir / "air_quality_train.csv"
        test_file = output_dir / "air_quality_test.csv"

        df_clean.to_csv(clean_file, index=False)
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print("💾 Archivos guardados:")
        print(f"   - Datos limpios: {clean_file}")
        print(f"   - Entrenamiento: {train_file}")
        print(f"   - Prueba: {test_file}")
        print(f"   - Columnas numéricas: {df_clean.select_dtypes(include=[np.number]).columns.tolist()}")
    else:
        print(f"❌ Fuente '{args.source}' no implementada aún")


if __name__ == "__main__":
    main()
