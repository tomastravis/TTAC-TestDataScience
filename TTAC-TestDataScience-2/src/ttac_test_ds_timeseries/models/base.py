"""Clase base para modelos de forecasting."""

import json
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=FutureWarning)


class BaseForecaster(ABC):
    """Clase base abstracta para modelos de forecasting."""

    def __init__(self, model_type: str, random_state: int = 42):
        """Inicializar forecaster base.

        Args:
            model_type: Tipo de modelo
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None
        self.metadata = {}

    @abstractmethod
    def fit(self, data, target_column: str = None):
        """Entrenar el modelo (método abstracto)."""
        pass

    @abstractmethod
    def predict(self, steps: int = 100) -> np.ndarray:
        """Realizar predicciones (método abstracto)."""
        pass

    def save_model(self, filepath: str):
        """Guardar modelo entrenado."""
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de guardar")

        model_data = {
            "model_type": self.model_type,
            "fitted_model": self.fitted_model,
            "metadata": self.metadata,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "random_state": self.random_state
        }

        # Guardar con joblib
        joblib.dump(model_data, filepath)

        # Guardar metadata separadamente
        metadata_path = str(filepath).replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str):
        """Cargar modelo guardado."""
        try:
            model_data = joblib.load(filepath)

            # Crear instancia
            instance = cls(model_type=model_data["model_type"],
                          random_state=model_data.get("random_state", 42))

            # Restaurar estado
            instance.fitted_model = model_data["fitted_model"]
            instance.metadata = model_data["metadata"]
            instance.feature_names = model_data["feature_names"]
            instance.is_fitted = model_data["is_fitted"]

            return instance

        except Exception as e:
            raise RuntimeError(f"Error cargando modelo: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "random_state": self.random_state
        }

    def _update_metadata(self, **kwargs):
        """Actualizar metadata del modelo."""
        self.metadata.update({
            "model_type": self.model_type,
            "fit_timestamp": datetime.now().isoformat(),
            **kwargs
        })
