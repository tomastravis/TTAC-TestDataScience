"""Utilidades simplificadas para análisis de series temporales."""

from typing import Any

import numpy as np
import pandas as pd


def calculate_time_series_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> dict[str, float]:
    """Calcular métricas para series temporales.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Diccionario con métricas calculadas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Evitar división por cero en MAPE
    mape = 0
    if len(y_true) > 0 and np.any(y_true != 0):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def create_sequences(data: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Crear secuencias para modelos LSTM.

    Args:
        data: Array de datos temporales
        sequence_length: Longitud de cada secuencia

    Returns:
        Tuple con (X, y) para entrenamiento
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)


def detect_seasonality(ts: pd.Series, freq: str = "D") -> dict[str, Any]:
    """Detectar estacionalidad en series temporales.

    Args:
        ts: Serie temporal
        freq: Frecuencia de los datos

    Returns:
        Información sobre estacionalidad detectada
    """
    try:

        result = {
            "has_seasonality": False,
            "weekly_pattern": False,
            "monthly_pattern": False,
            "seasonal_strength": 0.0
        }

        if len(ts) < 14:  # Datos insuficientes
            return result

        # Análisis básico de estacionalidad
        if freq == "D":
            # Estacionalidad semanal
            if hasattr(ts.index, "dayofweek"):
                weekly_groups = ts.groupby(ts.index.dayofweek).mean()
                weekly_var = weekly_groups.var()

                # Test de varianza significativa
                if weekly_var > ts.var() * 0.1:
                    result["weekly_pattern"] = True
                    result["has_seasonality"] = True

            # Estacionalidad mensual
            if hasattr(ts.index, "month"):
                monthly_groups = ts.groupby(ts.index.month).mean()
                monthly_var = monthly_groups.var()

                if monthly_var > ts.var() * 0.1:
                    result["monthly_pattern"] = True
                    result["has_seasonality"] = True

        # Calcular fuerza estacional simple
        if result["has_seasonality"]:
            result["seasonal_strength"] = min(weekly_var / ts.var() if "weekly_var" in locals() else 0,
                                            monthly_var / ts.var() if "monthly_var" in locals() else 0)

        return result

    except Exception as e:
        return {
            "has_seasonality": False,
            "error": str(e)
        }


def validate_air_quality_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validar estructura de datos de Air Quality UCI.

    Args:
        df: DataFrame con datos de Air Quality

    Returns:
        Resultado de validación
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "summary": {}
    }

    # Verificar columnas requeridas
    required_columns = ["CO(GT)"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Columnas faltantes: {missing_columns}")

    # Verificar tipos de datos
    if "CO(GT)" in df.columns and not pd.api.types.is_numeric_dtype(df["CO(GT)"]):
        validation_result["warnings"].append("CO(GT) no es numérico")

    # Resumen de datos
    validation_result["summary"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "columns": list(df.columns)
    }

    return validation_result


def load_air_quality_sample() -> pd.DataFrame:
    """Crear sample de datos Air Quality para testing.

    Returns:
        DataFrame con datos de muestra
    """
    # Crear datos de muestra basados en características reales del dataset UCI
    dates = pd.date_range("2004-03-01", periods=100, freq="H")

    # Simular CO(GT) con características realistas
    np.random.seed(42)
    co_base = 2.0  # mg/m³ promedio típico
    co_noise = np.random.normal(0, 0.5, len(dates))
    co_trend = np.linspace(0, 0.2, len(dates))  # Tendencia ligera

    sample_data = pd.DataFrame({
        "datetime": dates,
        "CO(GT)": co_base + co_trend + co_noise,
        "T": 15 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24),  # Temperatura
        "RH": 50 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24 + np.pi/4)  # Humedad
    })

    # Asegurar valores positivos para CO(GT)
    sample_data["CO(GT)"] = np.abs(sample_data["CO(GT)"])

    return sample_data
