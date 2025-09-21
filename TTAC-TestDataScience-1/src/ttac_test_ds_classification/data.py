"""Módulo simplificado para carga y procesamiento de datos."""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_wine_quality_dataset(include_both: bool = True) -> pd.DataFrame:
    """Cargar Wine Quality Dataset desde archivos locales."""
    try:
        # Intentar cargar desde data/raw/
        if include_both:
            df = pd.read_csv("data/raw/wine_quality_real.csv")
        else:
            df = pd.read_csv("data/raw/winequality-red.csv", sep=";")
            df["wine_type"] = "red"

        print(f"✓ Dataset cargado: {len(df)} muestras")
        return df
    except FileNotFoundError:
        print("Archivo de datos no encontrado en data/raw/")
        raise


def clean_wine_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar datos básico."""
    df_clean = df.copy()

    # Eliminar duplicados
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Eliminados {initial_len - len(df_clean)} duplicados")

    # Eliminar valores faltantes
    df_clean = df_clean.dropna()
    print(f"Dataset limpio: {len(df_clean)} muestras")

    return df_clean


def prepare_wine_features_target(
        df: pd.DataFrame, target_column: str = "quality") -> tuple[pd.DataFrame, pd.Series]:
    """Separar características y variable objetivo."""
    # Excluir columnas no numéricas y el target
    columns_to_exclude = [
        target_column,
        "wine_type"] if "wine_type" in df.columns else [target_column]

    X = df.drop(columns=columns_to_exclude)
    y = df[target_column]

    # Solo columnas numéricas
    X = X.select_dtypes(include=[np.number])

    return X, y


def split_wine_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42):
    """Dividir datos en entrenamiento y prueba."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y)
