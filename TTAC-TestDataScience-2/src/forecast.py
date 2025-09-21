"""Script de predicción para modelos de forecasting."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from ttac_test_ds_timeseries.data import load_air_quality_data
from ttac_test_ds_timeseries.models import AirQualityForecaster


def load_trained_model(model_path: str) -> AirQualityForecaster:
    """Cargar modelo entrenado."""
    print(f"📂 Cargando modelo desde: {model_path}")

    model = AirQualityForecaster.load_model(model_path)

    print(f"✅ Modelo {model.model_type.upper()} cargado exitosamente")
    return model


def make_predictions(model: AirQualityForecaster, steps: int = 100) -> np.ndarray:
    """Hacer predicciones con el modelo."""
    print(f"🔮 Generando {steps} predicciones...")

    predictions = model.predict(steps=steps)

    print(f"✅ Predicciones generadas: {len(predictions)} valores")
    return predictions


def save_predictions(predictions: np.ndarray, output_file: str, model_name: str):
    """Guardar predicciones a archivo."""
    # Crear DataFrame con predicciones
    results_df = pd.DataFrame({
        "step": range(1, len(predictions) + 1),
        "prediction": predictions,
        "model": model_name
    })

    # Guardar a CSV
    results_df.to_csv(output_file, index=False)
    print(f"💾 Predicciones guardadas en: {output_file}")

    # Mostrar estadísticas
    print("📊 Estadísticas de predicciones:")
    print(f"   Min: {predictions.min():.4f}")
    print(f"   Max: {predictions.max():.4f}")
    print(f"   Mean: {predictions.mean():.4f}")
    print(f"   Std: {predictions.std():.4f}")


def predict_with_historical_data(
    model_path: str,
    historical_data: pd.DataFrame = None,
    steps: int = 100
) -> np.ndarray:
    """Realizar predicción usando datos históricos."""
    # Cargar modelo
    model = load_trained_model(model_path)

    # Si no hay datos históricos, usar datos por defecto
    if historical_data is None:
        print("🔄 Cargando datos históricos de Air Quality UCI...")
        historical_data = load_air_quality_data(save_to_disk=False)

    # Hacer predicciones
    predictions = make_predictions(model, steps)

    return predictions


def main():
    """Función principal de predicción."""
    parser = argparse.ArgumentParser(description="Realizar predicciones con modelos entrenados")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Ruta al modelo entrenado (.pkl)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Número de pasos a predecir")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Archivo de salida para predicciones")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Ruta a datos históricos (opcional)")

    args = parser.parse_args()

    print("🔮 GENERACIÓN DE PREDICCIONES")
    print("=" * 40)

    try:
        # Verificar que el modelo existe
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {args.model_path}")

        # Cargar datos históricos si se especifican
        historical_data = None
        if args.data_path:
            data_path = Path(args.data_path)
            if data_path.exists():
                historical_data = pd.read_csv(args.data_path)
                print(f"📊 Datos históricos cargados: {historical_data.shape}")

        # Realizar predicciones
        predictions = predict_with_historical_data(
            str(model_path),
            historical_data,
            args.steps
        )

        # Extraer nombre del modelo del archivo
        model_name = model_path.stem.replace("_model", "")

        # Guardar resultados
        save_predictions(predictions, args.output, model_name)

        print("\n🎯 PREDICCIÓN COMPLETADA:")
        print(f"   Modelo: {model_name}")
        print(f"   Pasos: {args.steps}")
        print(f"   Archivo: {args.output}")

    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
