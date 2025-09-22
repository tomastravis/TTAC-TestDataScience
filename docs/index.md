# TTAC Data Science Tests

Este sitio documenta la implementación de dos proyectos completos de Machine Learning utilizando datasets reales del UCI ML Repository, desarrollados como parte de las pruebas técnicas de ciencia de datos.

## Proyectos Implementados

### TEST 1: Clasificación de Calidad de Vinos

**Objetivo**: Desarrollar un sistema de clasificación para predecir la calidad de vinos basado en sus propiedades fisicoquímicas.

- **Dataset**: UCI Wine Quality (6,497 muestras reales)
- **Tipo**: Problema de clasificación multiclase
- **Clases**: Calidad del vino (escala 3-9)
- **Features**: 11 propiedades fisicoquímicas
- **Modelos implementados**: Random Forest, SVM, XGBoost
- **Mejor resultado**: Random Forest con 69.31% de accuracy

### TEST 2: Forecasting de Series Temporales

**Objetivo**: Implementar modelos de predicción temporal para datos ambientales sin componente estacional.

- **Dataset**: UCI Air Quality (9,358 registros horarios)
- **Tipo**: Forecasting de series temporales
- **Variable objetivo**: CO(GT) - Concentración de monóxido de carbono
- **Horizonte**: 100 períodos de predicción
- **Modelos implementados**: ARIMA, LSTM, Prophet
- **Mejor resultado**: ARIMA con 21.84% MAPE

## Arquitectura del Proyecto

Ambos proyectos siguen una arquitectura modular y profesional:

```
TTAC-TestDataScience/
├── TTAC-TestDataScience-1/          # Proyecto de clasificación
│   ├── src/                         # Código fuente
│   ├── notebooks/                   # Análisis exploratorio y modelado
│   ├── tests/                       # Tests automatizados
│   └── data/                        # Datasets y resultados
├── TTAC-TestDataScience-2/          # Proyecto de series temporales
│   ├── src/                         # Código fuente
│   ├── notebooks/                   # Análisis temporal y forecasting
│   ├── tests/                       # Tests automatizados
│   └── data/                        # Datasets y modelos
└── docs/                            # Documentación unificada
```

## Características Principales

### Calidad del Código
- **Arquitectura modular**: Separación clara de responsabilidades
- **Type hints**: Código Python tipado para mejor mantenibilidad
- **Testing automatizado**: 36 tests implementados con pytest
- **Documentación**: Docstrings completas y notebooks autodocumentados

### Metodología de Desarrollo
- **EDA exhaustivo**: Análisis exploratorio detallado de ambos datasets
- **Validación robusta**: Cross-validation y métricas apropiadas para cada problema
- **Reproducibilidad**: Seeds fijos, entornos controlados, pipelines automatizados
- **CLI tools**: Scripts de línea de comandos para entrenamiento y predicción

### Stack Tecnológico
- **Python 3.12**: Lenguaje principal
- **Machine Learning**: scikit-learn, statsmodels, TensorFlow
- **Data Analysis**: pandas, numpy, matplotlib, seaborn
- **Testing**: pytest, mypy
- **Documentation**: MkDocs, Jupyter notebooks

## Resultados Destacados

Ambos proyectos cumplen con los requerimientos técnicos establecidos y demuestran implementaciones profesionales de sistemas de Machine Learning end-to-end:

- **TEST 1**: Accuracy del 69.31% en clasificación de calidad de vinos
- **TEST 2**: MAPE del 21.84% en forecasting de 100 períodos
- **Compliance**: Uso de datasets UCI reales, no financieros, con validación robusta

## Navegación

Utiliza el menú superior para explorar la documentación detallada de cada proyecto:

- **TEST 1**: Documentación completa del proyecto de clasificación
- **TEST 2**: Documentación completa del proyecto de series temporales  
- **Setup & Usage**: Instrucciones para ejecutar los proyectos

Cada sección incluye análisis de datasets, metodología, resultados y referencias de API para facilitar la comprensión y reproducción de los experimentos.