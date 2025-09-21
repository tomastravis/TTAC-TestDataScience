"""Módulo simplificado para modelos de machine learning."""


import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


class WineQualityClassifier:
    """Clasificador simplificado para calidad de vinos."""

    def __init__(
            self,
            model_type: str = "random_forest",
            random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None

        # Inicializar modelo
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=random_state
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrenar modelo."""
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"✓ Modelo {self.model_type} entrenado")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hacer predicciones."""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Obtener probabilidades de predicción."""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame,
                 y_test: pd.Series) -> dict[str, float]:
        """Evaluar modelo."""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado")

        y_pred = self.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted")
        }

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Obtener importancia de características."""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado")

        if hasattr(self.model, "feature_importances_"):
            return pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_
            }).sort_values("importance", ascending=False)
        else:
            print(f"Importancia no disponible para {self.model_type}")
            return None

    def save_model(self, filepath: str) -> None:
        """Guardar modelo."""
        if not self.is_fitted:
            raise ValueError("El modelo debe estar entrenado")

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }

        joblib.dump(model_data, filepath)
        print(f"✓ Modelo guardado en {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Cargar modelo."""
        model_data = joblib.load(filepath)

        instance = cls(model_type=model_data["model_type"])
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = model_data["is_fitted"]

        print(f"✓ Modelo cargado desde {filepath}")
        return instance


def compare_models(X_train, y_train, X_test,
                   y_test) -> dict[str, dict[str, float]]:
    """Comparar diferentes modelos."""
    models = ["random_forest", "svm", "xgboost"]
    results = {}

    for model_type in models:
        print(f"\n🤖 Entrenando {model_type}...")

        classifier = WineQualityClassifier(model_type=model_type)
        classifier.fit(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        results[model_type] = metrics

        print(
            f"Accuracy: {
                metrics['accuracy']:.4f}, F1: {
                metrics['f1_weighted']:.4f}")

    return results
