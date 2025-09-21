"""Módulo de modelos de forecasting para series temporales.

Este módulo proporciona múltiples modelos de forecasting:
- ARIMAForecaster: Modelo ARIMA tradicional
- ProphetForecaster: Modelo Prophet de Facebook  
- LSTMForecaster: Redes neuronales LSTM
- LinearForecaster: Regresión lineal simple
- AirQualityForecaster: Wrapper unificado (compatibilidad)
"""

# Importar modelos específicos
from .arima_model import ARIMAForecaster
from .prophet_model import ProphetForecaster  
from .lstm_model import LSTMForecaster
from .linear_model import LinearForecaster
from .base import BaseForecaster

# Importar la clase original para compatibilidad
import json
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)


class AirQualityForecaster:
    """Forecaster unificado para Air Quality UCI dataset (compatibilidad)."""

    def __init__(self, model_type: str = "arima", **kwargs):
        """Inicializar forecaster unificado.

        Args:
            model_type: Tipo de modelo ('arima', 'prophet', 'lstm', 'linear')
            **kwargs: Argumentos específicos del modelo
        """
        self.model_type = model_type
        
        # Crear modelo específico según tipo
        if model_type == "arima":
            order = kwargs.get('order', (1, 1, 1))
            self.model = ARIMAForecaster(order=order, **{k: v for k, v in kwargs.items() if k != 'order'})
        elif model_type == "prophet":
            self.model = ProphetForecaster(**kwargs)
        elif model_type == "lstm":
            self.model = LSTMForecaster(**kwargs)
        elif model_type == "linear":
            self.model = LinearForecaster(**kwargs)
        else:
            raise ValueError(f"Modelo no soportado: {model_type}. Use: arima, prophet, lstm, linear")

    def fit(self, data, target_column: str = "CO(GT)"):
        """Entrenar el modelo."""
        return self.model.fit(data, target_column)

    def predict(self, steps: int = 100) -> np.ndarray:
        """Realizar predicciones."""
        return self.model.predict(steps)

    def save_model(self, filepath: str):
        """Guardar modelo entrenado."""
        return self.model.save_model(filepath)

    @classmethod
    def load_model(cls, filepath: str):
        """Cargar modelo guardado."""
        try:
            model_data = joblib.load(filepath)
            model_type = model_data.get("model_type", "linear")
            
            # Crear instancia wrapper
            instance = cls(model_type=model_type)
            
            # Cargar modelo específico
            if model_type == "arima":
                instance.model = ARIMAForecaster.load_model(filepath)
            elif model_type == "prophet":
                instance.model = ProphetForecaster.load_model(filepath)
            elif model_type == "lstm":
                instance.model = LSTMForecaster.load_model(filepath)
            elif model_type == "linear":
                instance.model = LinearForecaster.load_model(filepath)
            
            return instance
            
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo: {e}")

    def get_model_info(self) -> dict[str, Any]:
        """Obtener información del modelo."""
        return self.model.get_model_info()

    @property
    def is_fitted(self) -> bool:
        """Verificar si el modelo está entrenado."""
        return self.model.is_fitted

    @property
    def fitted_model(self):
        """Acceder al modelo entrenado."""
        return self.model.fitted_model

    @property
    def metadata(self) -> dict:
        """Acceder a metadata del modelo."""
        return self.model.metadata

    @property
    def feature_names(self):
        """Acceder a nombres de features."""
        return self.model.feature_names


# Exportar todas las clases
__all__ = [
    "BaseForecaster",
    "ARIMAForecaster", 
    "ProphetForecaster",
    "LSTMForecaster", 
    "LinearForecaster",
    "AirQualityForecaster"  # Para compatibilidad
]
