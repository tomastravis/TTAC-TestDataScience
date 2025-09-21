"""TTAC Test Data Science - Clasificación

TEST 1 - Clasificación: Wine Quality Dataset con Random Forest, SVM, XGBoost

Autor: Tomás Travis Alonso Cremnitz
Email: tomasnataliaalbanes@gmail.com
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Tomás Travis Alonso Cremnitz"
__email__ = "tomasnataliaalbanes@gmail.com"

# Importar las clases principales para facilitar el uso
from .data import (
    clean_wine_data,
    load_wine_quality_dataset,
    prepare_wine_features_target,
)
from .models import WineQualityClassifier, compare_models

__all__ = [
    "WineQualityClassifier",
    "compare_models",
    "load_wine_quality_dataset",
    "clean_wine_data",
    "prepare_wine_features_target"
]
