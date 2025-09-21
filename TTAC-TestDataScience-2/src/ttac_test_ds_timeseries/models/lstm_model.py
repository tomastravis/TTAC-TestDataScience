"""Modelo LSTM para forecasting de series temporales."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .base import BaseForecaster


class LSTMForecaster(BaseForecaster):
    """Forecaster basado en redes LSTM."""

    def __init__(self, 
                 sequence_length: int = 100,
                 units: int = 50,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """Inicializar modelo LSTM.

        Args:
            sequence_length: Longitud de secuencia de entrada
            units: Número de unidades LSTM
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño de batch
            validation_split: Proporción para validación
            random_state: Semilla para reproducibilidad
        """
        super().__init__("lstm", random_state)
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.scaler = None
        self._init_lstm_model()

    def _init_lstm_model(self):
        """Inicializar componentes LSTM."""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            
            # Configurar semilla para reproducibilidad
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
            
            self.tf = tf
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            
        except ImportError:
            raise ImportError("tensorflow y scikit-learn requeridos para LSTM")

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crear secuencias para LSTM.

        Args:
            data: Datos escalados

        Returns:
            Tuple con (X, y) para entrenamiento
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]):
        """Construir arquitectura LSTM.

        Args:
            input_shape: Forma de entrada (timesteps, features)

        Returns:
            Modelo compilado de Keras
        """
        model = self.tf.keras.Sequential([
            self.tf.keras.layers.LSTM(
                self.units, 
                return_sequences=True, 
                input_shape=input_shape
            ),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.LSTM(self.units, return_sequences=False),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model

    def fit(self, data, target_column: str = "CO(GT)"):
        """Entrenar modelo LSTM.

        Args:
            data: DataFrame o Series con datos de entrenamiento
            target_column: Columna objetivo
        """
        # Convertir a Series si es necesario
        if isinstance(data, pd.Series):
            ts_data = data
            if hasattr(data, 'name') and data.name:
                target_column = data.name
        else:
            if target_column not in data.columns:
                raise ValueError(f"Columna {target_column} no encontrada en datos")
            ts_data = data[target_column]

        # Limpiar y preparar datos
        ts = ts_data.dropna().values.reshape(-1, 1)
        
        if len(ts) < self.sequence_length + 100:
            raise ValueError(f"Necesario al menos {self.sequence_length + 100} datos para LSTM")

        try:
            # Escalar datos
            scaled_data = self.scaler.fit_transform(ts)
            
            # Crear secuencias
            X, y = self._create_sequences(scaled_data)
            
            # Reshape para LSTM (samples, timesteps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Construir modelo
            self.model = self._build_model((self.sequence_length, 1))
            
            # Entrenar modelo
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0  # Silencioso por defecto
            )
            
            # Guardar modelo entrenado
            self.fitted_model = {
                'model': self.model,
                'scaler': self.scaler,
                'history': history.history,
                'last_sequence': scaled_data[-self.sequence_length:]
            }
            
            self.is_fitted = True
            self.feature_names = [target_column]

            # Actualizar metadata
            self._update_metadata(
                target_column=target_column,
                training_samples=len(ts),
                sequence_length=self.sequence_length,
                units=self.units,
                epochs=self.epochs,
                final_loss=history.history['loss'][-1],
                final_val_loss=history.history.get('val_loss', [None])[-1]
            )

        except Exception as e:
            raise RuntimeError(f"Error entrenando LSTM: {e}")

        return self

    def predict(self, steps: int = 100) -> np.ndarray:
        """Realizar predicciones LSTM.

        Args:
            steps: Número de períodos a predecir

        Returns:
            Array con predicciones
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de predecir")

        try:
            model = self.fitted_model['model']
            scaler = self.fitted_model['scaler']
            last_sequence = self.fitted_model['last_sequence'].copy()
            
            predictions = []
            current_sequence = last_sequence.reshape(1, self.sequence_length, 1)
            
            # Generar predicciones paso a paso
            for _ in range(steps):
                # Predecir siguiente valor
                next_pred = model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Actualizar secuencia
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Desescalar predicciones
            predictions = np.array(predictions).reshape(-1, 1)
            predictions_unscaled = scaler.inverse_transform(predictions)
            
            return predictions_unscaled.flatten()
            
        except Exception as e:
            raise RuntimeError(f"Error prediciendo LSTM: {e}")

    def get_training_history(self):
        """Obtener historial de entrenamiento.

        Returns:
            Dict con métricas de entrenamiento
        """
        if not self.is_fitted:
            return {"error": "Modelo no entrenado"}

        return self.fitted_model.get('history', {})

    def plot_training_history(self):
        """Generar gráfico del historial de entrenamiento.

        Returns:
            Figure de matplotlib
        """
        if not self.is_fitted:
            raise ValueError("Modelo debe ser entrenado antes de graficar")

        try:
            import matplotlib.pyplot as plt
            
            history = self.get_training_history()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Gráfico de pérdida
            ax1.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Gráfico de MAE
            if 'mae' in history:
                ax2.plot(history['mae'], label='Training MAE')
                if 'val_mae' in history:
                    ax2.plot(history['val_mae'], label='Validation MAE')
                ax2.set_title('Model MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True)
            else:
                ax2.text(0.5, 0.5, 'MAE not available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Error graficando historial: {e}")
