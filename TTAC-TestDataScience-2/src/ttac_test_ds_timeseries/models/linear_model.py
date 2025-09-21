"""Modelo lineal para forecasting de series temporales."""

import numpy as np
import pandas as pd
from typing import Optional

from .base import BaseForecaster


class LinearForecaster(BaseForecaster):
    """Forecaster basado en regresión lineal simple."""

    def __init__(self, random_state: int = 42):
        """Inicializar modelo lineal.

        Args:
            random_state: Semilla para reproducibilidad
        """
        super().__init__("linear", random_state)
        self._init_linear_model()

    def _init_linear_model(self):
        """Inicializar modelo lineal."""
        try:
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        except ImportError:
            raise ImportError("scikit-learn requerido para modelo lineal")

    def fit(self, data, target_column: str = "CO(GT)"):
        """Entrenar modelo lineal.

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
            # Crear features temporales simples
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values

            # Entrenar modelo lineal
            self.fitted_model = self.model.fit(X, y)
            self.is_fitted = True
            self.feature_names = [target_column]

            # Calcular métricas básicas
            y_pred = self.fitted_model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            r2 = self.fitted_model.score(X, y)

            # Actualizar metadata
            self._update_metadata(
                target_column=target_column,
                training_samples=len(ts),
                r2_score=r2,
                mse=mse,
                slope=self.fitted_model.coef_[0],
                intercept=self.fitted_model.intercept_
            )

        except Exception as e:
            raise RuntimeError(f"Error entrenando modelo lineal: {e}")

        return self

    def predict(self, steps: int = 100) -> np.ndarray:
        """Realizar predicciones lineales.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Array con predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            # Obtener último índice de entrenamiento
            last_index = self.metadata.get('training_samples', 0)
            
            # Crear índices futuros
            future_X = np.arange(last_index, last_index + steps).reshape(-1, 1)
            
            # Predecir
            predictions = self.fitted_model.predict(future_X)
            
            return predictions
        except Exception as e:
            raise RuntimeError(f"Error prediciendo modelo lineal: {e}")

    def predict_with_trend_analysis(self, steps: int = 100):
        """Realizar predicciones con análisis de tendencia.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Dict con predicciones y análisis de tendencia
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            predictions = self.predict(steps)
            
            # Análisis de tendencia
            slope = self.metadata.get('slope', 0)
            trend_direction = "creciente" if slope > 0 else "decreciente" if slope < 0 else "estable"
            
            # Calcular cambio total en el período
            total_change = slope * steps
            avg_change_per_period = slope
            
            return {
                "predictions": predictions,
                "trend_analysis": {
                    "direction": trend_direction,
                    "slope": slope,
                    "total_change_over_period": total_change,
                    "avg_change_per_period": avg_change_per_period,
                    "r2_score": self.metadata.get('r2_score', None)
                }
            }
        except Exception as e:
            raise RuntimeError(f"Error en análisis de tendencia: {e}")

    def get_linear_equation(self) -> str:
        """Obtener ecuación lineal en formato legible.

        Returns:
            String con la ecuación lineal
        """
        if not self.is_fitted:
            return "Modelo no entrenado"

        try:
            slope = self.metadata.get('slope', 0)
            intercept = self.metadata.get('intercept', 0)
            
            sign = "+" if intercept >= 0 else "-"
            abs_intercept = abs(intercept)
            
            return f"y = {slope:.6f}x {sign} {abs_intercept:.6f}"
        except Exception as e:
            return f"Error obteniendo ecuación: {e}"

    def plot_linear_fit(self, original_data: pd.Series = None):
        """Generar gráfico del ajuste lineal.

        Args:
            original_data: Datos originales para comparación

        Returns:
            Figure de matplotlib
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de graficar")

        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Si se proporcionan datos originales, graficarlos
            if original_data is not None:
                ax.plot(original_data.values, 'b-', alpha=0.7, label='Datos originales')
                
                # Crear ajuste lineal sobre los datos originales
                X_fit = np.arange(len(original_data)).reshape(-1, 1)
                y_fit = self.fitted_model.predict(X_fit)
                ax.plot(y_fit, 'r-', linewidth=2, label='Ajuste lineal')
            
            # Información del modelo
            equation = self.get_linear_equation()
            r2 = self.metadata.get('r2_score', 0)
            
            ax.set_title(f'Ajuste Lineal: {equation}\nR² = {r2:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Valor')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Error graficando ajuste lineal: {e}")
