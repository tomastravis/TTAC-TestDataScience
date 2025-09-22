# TEST 1: Wine Quality Classification

## Descripción del Proyecto

El TEST 1 implementa un sistema completo de clasificación de Machine Learning para predecir la calidad de vinos basándose en sus propiedades fisicoquímicas. Este proyecto utiliza el dataset Wine Quality del UCI ML Repository y evalúa múltiples algoritmos de clasificación.

## Objetivos

- **Objetivo principal**: Clasificar la calidad de vinos en una escala de 3 a 9
- **Objetivo técnico**: Comparar el rendimiento de diferentes algoritmos de clasificación
- **Objetivo metodológico**: Implementar un pipeline completo de Machine Learning

## Dataset Utilizado

**Fuente**: UCI ML Repository - Wine Quality Dataset  
**Tipo**: Problema de clasificación multiclase  
**Tamaño**: 6,497 muestras totales

### Composición del Dataset
- **Vinos tintos**: 1,599 muestras (24.6%)
- **Vinos blancos**: 4,898 muestras (75.4%)
- **Features**: 11 propiedades fisicoquímicas
- **Target**: Calidad (valores entre 3 y 9)

### Distribución de Clases
```
Calidad 3:    30 muestras (0.5%)
Calidad 4:   216 muestras (3.3%)
Calidad 5: 2,138 muestras (32.9%)
Calidad 6: 2,836 muestras (43.7%)
Calidad 7: 1,079 muestras (16.6%)
Calidad 8:   193 muestras (3.0%)
Calidad 9:     5 muestras (0.1%)
```

## Arquitectura del Proyecto

### Estructura de Directorios
```
TTAC-TestDataScience-1/
├── src/
│   ├── ttac_test_ds_classification/
│   │   ├── data.py          # Manejo de datos
│   │   └── models.py        # Modelos de ML
│   ├── train_model.py       # Script de entrenamiento
│   └── inference.py         # Script de predicción
├── notebooks/
│   ├── 01_eda_classification.ipynb
│   └── 02_modeling_classification.ipynb
├── tests/
│   ├── test_process.py
│   └── test_train_model.py
└── data/
    ├── raw/
    ├── processed/
    └── final/
```

### Componentes Principales

#### 1. Módulo de Datos (`data.py`)
- Carga y validación del dataset UCI
- Preprocesamiento y limpieza de datos
- División estratificada train/test
- Escalado de características

#### 2. Módulo de Modelos (`models.py`)
- Implementación de múltiples algoritmos:
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- Validación cruzada estratificada
- Métricas de evaluación especializadas

#### 3. Scripts CLI
- **`train_model.py`**: Entrenamiento automatizado
- **`inference.py`**: Predicciones en nuevos datos

## Metodología Implementada

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis estadístico descriptivo completo
- Visualización de distribuciones por tipo de vino
- Matriz de correlaciones entre features
- Detección de outliers y valores faltantes

### 2. Preprocesamiento
- Tratamiento de valores faltantes (ninguno encontrado)
- Escalado StandardScaler para features numéricas
- División estratificada 80/20 train/test
- Validación de distribuciones balanceadas

### 3. Modelado y Evaluación
- Implementación de 3 algoritmos de clasificación
- Validación cruzada estratificada (5-fold)
- Métricas específicas para clases desbalanceadas:
  - Accuracy
  - F1-Score weighted
  - Precision y Recall por clase
  - Matriz de confusión

## Principales Hallazgos

### Características del Dataset
- **Desbalance significativo**: Concentración en calidades medias (5-7)
- **Clases extremas minoritarias**: Calidades 3 y 9 representan < 1%
- **Diferencias químicas**: Vinos tintos y blancos tienen perfiles distintos
- **Variables más importantes**: Alcohol, acidez volátil, sulfatos

### Insights del EDA
- El alcohol es el predictor más fuerte de calidad
- Los vinos blancos dominan el dataset (75.4%)
- Las calidades extremas (3, 9) son muy raras
- Existe correlación entre acidez y calidad percibida

## Navegación

- **[Dataset & EDA](dataset.md)**: Análisis detallado del dataset y exploración
- **[Models & Results](models.md)**: Implementación de modelos y resultados
- **[API Reference](api.md)**: Documentación técnica del código