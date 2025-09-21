"""Training script for Air Quality forecasting models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from ttac_test_ds_timeseries.data import (
    clean_timeseries_data,
    load_air_quality_data,
)
from ttac_test_ds_timeseries.models import AirQualityForecaster
from utils import calculate_time_series_metrics


def load_timeseries_data(data_source: str, **kwargs) -> pd.DataFrame:
    """Cargar datos de series temporales."""
    print(f"🔄 Cargando datos de {data_source}...")

    if data_source == "air_quality":
        target_column = kwargs.get("target", "CO(GT)")
        data = load_air_quality_data(
            save_to_disk=True,
            target_column=target_column)
    else:
        raise ValueError(
            f"Fuente de datos no soportada: {data_source}. Use 'air_quality'.")

    print(f"✅ Datos cargados: {len(data)} registros")
    return data


def prepare_data_for_training(data: pd.DataFrame, test_size: float = 0.2, target_column: str = None):
    """Preparar datos para entrenamiento."""
    # Limpiar datos
    data_clean = clean_timeseries_data(data)

    # Asegurar columna de fecha
    date_column = "datetime" if "datetime" in data.columns else "date"
    if date_column not in data.columns:
        data[date_column] = pd.date_range(
            start="2004-03-10", periods=len(data), freq="H")

    # Determinar columna objetivo
    if target_column and target_column in data.columns:
        target_col = target_column
    else:
        # Usar la primera columna numérica
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No se encontraron columnas numéricas en los datos")
        target_col = numeric_columns[0]

    print(f"🎯 Columna objetivo: {target_col}")

    # Crear serie temporal
    if date_column in data_clean.columns:
        ts = data_clean.set_index(date_column)[target_col]
    else:
        ts = data_clean[target_col]

    # Dividir datos
    split_point = int(len(ts) * (1 - test_size))
    train_data = ts[:split_point]
    test_data = ts[split_point:]

    print(f"📊 División: Train={len(train_data)}, Test={len(test_data)}")

    return {
        "train": train_data,
        "test": test_data,
        "full": ts,
        "target_column": target_col
    }


def train_forecasting_model(model_type: str, train_data: pd.Series, **kwargs):
    """Entrenar modelo de forecasting."""
    print(f"🚀 Entrenando modelo {model_type.upper()}...")

    # Crear modelo
    forecaster = AirQualityForecaster(model_type=model_type)

    # Entrenar
    forecaster.fit(train_data)

    print(f"✅ Modelo {model_type.upper()} entrenado exitosamente")
    return forecaster


def evaluate_model(model, test_data: pd.Series, horizon: int = 100):
    """Evaluar modelo de forecasting."""
    print("📈 Evaluando modelo...")

    try:
        # Realizar predicciones
        steps = min(horizon, len(test_data))
        predictions = model.predict(steps=steps)

        # Calcular métricas
        actual = test_data[:steps].values
        metrics = calculate_time_series_metrics(actual, predictions)

        print("📊 Métricas de evaluación:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")

        return metrics

    except Exception as e:
        print(f"❌ Error en evaluación: {e}")
        return {}


def save_model_results(model, metrics: dict, output_dir: str, model_name: str):
    """Guardar modelo y resultados."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    model_file = output_path / f"{model_name}_model.pkl"
    model.save_model(str(model_file))

    # Guardar métricas
    metrics_file = output_path / f"{model_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"💾 Resultados guardados en: {output_dir}")
    return model_file, metrics_file


def main():
    """Función principal de entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenar modelos de forecasting")
    parser.add_argument("--data", type=str, default="air_quality",
                        help="Fuente de datos (air_quality)")
    parser.add_argument("--target", type=str, default="CO(GT)",
                        help="Columna objetivo")
    parser.add_argument("--models", nargs="+", default=["linear", "arima"],
                        help="Modelos a entrenar")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directorio de salida")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proporción de datos para prueba")
    parser.add_argument("--horizon", type=int, default=100,
                        help="Horizonte de predicción")

    args = parser.parse_args()

    print("🎯 ENTRENAMIENTO DE MODELOS DE FORECASTING")
    print("=" * 50)

    try:
        # Cargar datos
        data = load_timeseries_data(
            args.data,
            target=args.target,
            location="UCI Air Quality"
        )

        # Preparar datos
        prepared_data = prepare_data_for_training(
            data,
            test_size=args.test_size,
            target_column=args.target
        )

        X_train = prepared_data["train"]
        X_test = prepared_data["test"]

        # Entrenar modelos
        all_metrics = {}
        models_to_train = args.models

        for model_type in models_to_train:
            print(f"\n🔄 Entrenando {model_type.upper()}...")

            try:
                if model_type == "arima":
                    model = train_forecasting_model("arima", X_train)
                elif model_type == "prophet":
                    model = train_forecasting_model("prophet", X_train)
                elif model_type == "linear":
                    model = train_forecasting_model("linear", X_train)
                else:
                    print(f"❌ Modelo {model_type} no soportado")
                    continue

                # Evaluar modelo
                metrics = evaluate_model(model, X_test, args.horizon)

                # Guardar resultados
                save_model_results(model, metrics, args.output_dir, model_type)

                all_metrics[model_type] = metrics

            except Exception as e:
                print(f"❌ Error entrenando {model_type}: {e}")
                continue

        # Resumen final
        print("\n🏆 RESUMEN DE ENTRENAMIENTO:")
        for model_type, metrics in all_metrics.items():
            if metrics:
                print(
                    f"{model_type.upper():>10}: MAE={metrics.get('MAE', 'N/A'):.4f}, "
                    f"RMSE={metrics.get('RMSE', 'N/A'):.4f}, "
                    f"MAPE={metrics.get('MAPE', 'N/A'):.2f}%"
                )

        print(f"\n💾 Modelos guardados en: {args.output_dir}")

    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
