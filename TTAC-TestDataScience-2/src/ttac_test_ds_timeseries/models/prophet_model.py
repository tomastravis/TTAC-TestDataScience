"""Modelo Prophet para forecasting de series temporales."""

import numpy as np
import pandas as pd
from typing import Optional

from .base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Forecaster basado en Prophet de Facebook."""

    def __init__(self, 
                 yearly_seasonality: bool = False,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 random_state: int = 42):
        """Inicializar modelo Prophet.

        Args:
            yearly_seasonality: Incluir estacionalidad anual
            weekly_seasonality: Incluir estacionalidad semanal  
            daily_seasonality: Incluir estacionalidad diaria
            random_state: Semilla para reproducibilidad
        """
        super().__init__("prophet", random_state)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self._init_prophet_model()

    def _init_prophet_model(self):
        """Inicializar modelo Prophet."""
        try:
            from prophet import Prophet
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
        except ImportError:
            raise ImportError("prophet requerido para Prophet")

    def fit(self, data, target_column: str = "CO(GT)"):
        """Entrenar modelo Prophet.

        Args:
            data: DataFrame con datos de entrenamiento
            target_column: Columna objetivo
        """
        # Prophet requiere formato específico
        if isinstance(data, pd.Series):
            if not hasattr(data, 'index') or not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Series debe tener índice datetime para Prophet")
            
            prophet_data = pd.DataFrame({
                "ds": data.index,
                "y": data.values
            })
            target_column = data.name or target_column
        else:
            if target_column not in data.columns:
                raise ValueError(f"Columna {target_column} no encontrada en datos")
            
            # Buscar columna de fecha
            date_column = None
            for col in ["datetime", "date", "ds"]:
                if col in data.columns:
                    date_column = col
                    break
            
            if date_column is None:
                # Usar índice si es datetime
                if isinstance(data.index, pd.DatetimeIndex):
                    prophet_data = pd.DataFrame({
                        "ds": data.index,
                        "y": data[target_column]
                    })
                else:
                    raise ValueError("No se encontró columna de fecha para Prophet")
            else:
                prophet_data = pd.DataFrame({
                    "ds": data[date_column],
                    "y": data[target_column]
                })

        # Limpiar datos
        prophet_data = prophet_data.dropna()

        try:
            # Entrenar modelo Prophet
            self.fitted_model = self.model.fit(prophet_data)
            self.is_fitted = True
            self.feature_names = [target_column]

            # Actualizar metadata
            self._update_metadata(
                target_column=target_column,
                training_samples=len(prophet_data),
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )

        except Exception as e:
            raise RuntimeError(f"Error entrenando Prophet: {e}")

        return self

    def predict(self, steps: int = 100) -> np.ndarray:
        """Realizar predicciones Prophet.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Array con predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            # Crear fechas futuras
            future = self.fitted_model.make_future_dataframe(periods=steps, freq="h")
            forecast = self.fitted_model.predict(future)
            
            # Retornar solo predicciones futuras
            return forecast["yhat"].tail(steps).values
        except Exception as e:
            raise RuntimeError(f"Error prediciendo Prophet: {e}")

    def predict_with_components(self, steps: int = 100):
        """Realizar predicciones con componentes de Prophet.

        Args:
            steps: Número de períodos a predecir

        Returns:
            DataFrame con predicciones y componentes
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            # Crear fechas futuras
            future = self.fitted_model.make_future_dataframe(periods=steps, freq="h")
            forecast = self.fitted_model.predict(future)
            
            # Retornar componentes principales
            components = ["yhat", "yhat_lower", "yhat_upper", "trend"]
            
            # Agregar componentes estacionales si existen
            if "weekly" in forecast.columns:
                components.append("weekly")
            if "daily" in forecast.columns:
                components.append("daily")
            if "yearly" in forecast.columns:
                components.append("yearly")
            
            return forecast[components].tail(steps)
        except Exception as e:
            raise RuntimeError(f"Error prediciendo componentes: {e}")

    def plot_forecast(self, steps: int = 100):
        """Generar gráfico de predicciones Prophet.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Figure de matplotlib
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de graficar")

        try:
            # Crear predicciones
            future = self.fitted_model.make_future_dataframe(periods=steps, freq="h")
            forecast = self.fitted_model.predict(future)
            
            # Crear gráfico
            fig = self.fitted_model.plot(forecast)
            fig.suptitle(f"Prophet Forecast - {steps} períodos", fontsize=14, fontweight='bold')
            
            return fig
        except Exception as e:
            raise RuntimeError(f"Error graficando Prophet: {e}")

    def plot_components(self, steps: int = 100):
        """Generar gráfico de componentes Prophet.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Figure de matplotlib
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de graficar")

        try:
            # Crear predicciones
            future = self.fitted_model.make_future_dataframe(periods=steps, freq="h")
            forecast = self.fitted_model.predict(future)
            
            # Crear gráfico de componentes
            fig = self.fitted_model.plot_components(forecast)
            fig.suptitle("Prophet Components", fontsize=14, fontweight='bold')
            
            return fig
        except Exception as e:
            raise RuntimeError(f"Error graficando componentes: {e}")
