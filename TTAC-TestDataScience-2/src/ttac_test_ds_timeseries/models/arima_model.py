"""Modelo ARIMA para forecasting de series temporales."""

import numpy as np
import pandas as pd
from typing import Optional

from .base import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """Forecaster basado en modelo ARIMA."""

    def __init__(self, order: tuple = (1, 1, 1), random_state: int = 42):
        """Inicializar modelo ARIMA.

        Args:
            order: Orden ARIMA (p, d, q)
            random_state: Semilla para reproducibilidad
        """
        super().__init__("arima", random_state)
        self.order = order
        self._init_arima_model()

    def _init_arima_model(self):
        """Inicializar modelo ARIMA."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA
        except ImportError:
            raise ImportError("statsmodels requerido para ARIMA")

    def fit(self, data, target_column: str = "CO(GT)"):
        """Entrenar modelo ARIMA.

        Args:
            data: DataFrame o Series con datos de entrenamiento
            target_column: Columna objetivo
        """
        # Convertir Series a datos utilizables
        if isinstance(data, pd.Series):
            ts_data = data
            if hasattr(data, 'name') and data.name:
                target_column = data.name
        else:
            if target_column not in data.columns:
                raise ValueError(f"Columna {target_column} no encontrada en datos")
            ts_data = data[target_column]

        # Limpiar datos
        ts = ts_data.dropna()

        try:
            # Entrenar modelo ARIMA
            self.fitted_model = self.model(ts, order=self.order).fit()
            self.is_fitted = True
            self.feature_names = [target_column]

            # Actualizar metadata
            self._update_metadata(
                target_column=target_column,
                training_samples=len(ts),
                arima_order=self.order,
                aic=self.fitted_model.aic,
                bic=self.fitted_model.bic
            )

        except Exception as e:
            raise RuntimeError(f"Error entrenando ARIMA: {e}")

        return self

    def predict(self, steps: int = 100) -> np.ndarray:
        """Realizar predicciones ARIMA.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Array con predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            raise RuntimeError(f"Error prediciendo ARIMA: {e}")

    def predict_with_intervals(self, steps: int = 100, alpha: float = 0.05):
        """Realizar predicciones con intervalos de confianza.

        Args:
            steps: Número de períodos a predecir
            alpha: Nivel de significancia (0.05 para 95% confianza)

        Returns:
            Dict con predicciones e intervalos
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=alpha)

            return {
                "forecast": np.array(forecast),
                "lower_bound": np.array(conf_int.iloc[:, 0]),
                "upper_bound": np.array(conf_int.iloc[:, 1]),
                "confidence_level": int((1 - alpha) * 100)
            }
        except Exception as e:
            raise RuntimeError(f"Error prediciendo con intervalos: {e}")

    def get_model_diagnostics(self):
        """Obtener diagnósticos del modelo ARIMA."""
        if not self.is_fitted:
            return {"error": "Modelo no entrenado"}

        try:
            return {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "hqic": self.fitted_model.hqic,
                "llf": self.fitted_model.llf,
                "order": self.order,
                "params": self.fitted_model.params.to_dict() if hasattr(self.fitted_model.params, 'to_dict') else {}
            }
        except Exception as e:
            return {"error": f"Error obteniendo diagnósticos: {e}"}
