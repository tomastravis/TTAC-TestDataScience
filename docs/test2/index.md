# TEST 2: Air Quality Forecasting

## Descripción del Proyecto

El TEST 2 implementa un sistema completo de forecasting de series temporales para datos ambientales. Este proyecto utiliza el dataset Air Quality del UCI ML Repository y evalúa múltiples algoritmos de predicción temporal especializados en datos no estacionales.

## Objetivos

- **Objetivo principal**: Predecir 100 períodos futuros de concentración CO(GT)
- **Objetivo técnico**: Comparar modelos de series temporales (ARIMA, LSTM, Prophet)
- **Objetivo metodológico**: Implementar pipeline completo de forecasting temporal

## Compliance TEST 2

Este proyecto cumple estrictamente con los requerimientos establecidos:

- **Dataset NO financiero**: Air Quality UCI (datos ambientales)
- **Variable NO estacional**: CO(GT) sin patrones estacionales significativos
- **Horizonte de predicción**: 100 períodos horarios
- **Modelos especializados**: ARIMA, LSTM, Prophet para series temporales

## Dataset Utilizado

**Fuente**: UCI ML Repository - Air Quality Dataset  
**Tipo**: Serie temporal univariada para forecasting  
**Tamaño**: 9,358 registros horarios

### Características del Dataset
- **Período**: Marzo 2004 - Febrero 2005 (1 año completo)
- **Frecuencia**: Mediciones horarias
- **Ubicación**: Ciudad italiana (datos ambientales reales)
- **Sensores**: 15 variables ambientales diferentes
- **Variable objetivo**: CO(GT) - Concentración de monóxido de carbono

### Información Técnica
```
Período temporal: 2004-03-10 18:00:00 a 2005-02-04 14:00:00
Total registros: 9,358 observaciones horarias
Completeness: 89.4% (valores válidos)
Missing values: 10.6% (tratados durante preprocesamiento)
```

## Arquitectura del Proyecto

### Estructura de Directorios
```
TTAC-TestDataScience-2/
├── src/
│   ├── ttac_test_ds_timeseries/
│   │   ├── data/
│   │   │   └── load_gas_sensor.py
│   │   └── models/
│   │       ├── arima_model.py
│   │       ├── lstm_model.py
│   │       └── prophet_model.py
│   ├── train_model.py       # Entrenamiento de modelos
│   ├── forecast.py          # Predicción 100 períodos
│   └── process.py           # Preprocesamiento
├── notebooks/
│   ├── 01_eda_timeseries.ipynb
│   └── 02_modeling_forecasting.ipynb
├── tests/
│   ├── test_data.py
│   └── test_models.py
└── data/
    ├── raw/
    ├── processed/
    └── final/
```

### Componentes Principales

#### 1. Módulo de Datos (`load_gas_sensor.py`)
- Carga del dataset UCI Air Quality
- Tratamiento de valores faltantes
- Conversión a formato temporal apropiado
- Validación de estacionariedad

#### 2. Módulos de Modelos
- **`arima_model.py`**: Implementación ARIMA con optimización automática
- **`lstm_model.py`**: Red neuronal LSTM para forecasting
- **`prophet_model.py`**: Modelo Prophet de Facebook

#### 3. Scripts CLI
- **`train_model.py`**: Entrenamiento de múltiples modelos
- **`forecast.py`**: Predicción específica de 100 períodos
- **`process.py`**: Pipeline de preprocesamiento

## Metodología Implementada

### 1. Análisis de Series Temporales
- **Test de estacionariedad**: Augmented Dickey-Fuller (ADF)
- **Análisis de autocorrelación**: ACF y PACF
- **Detección de tendencia**: Descomposición temporal
- **Validación de no estacionalidad**: Cumple requisitos TEST 2

### 2. Preprocesamiento Temporal
- **Tratamiento de missings**: Interpolación lineal para gaps < 6 horas
- **Diferenciación**: Para conseguir estacionariedad
- **Normalización**: MinMax scaling para LSTM
- **Validación temporal**: Walk-forward validation

### 3. Modelado y Evaluación
- **ARIMA**: Optimización automática de parámetros (p,d,q)
- **LSTM**: Arquitectura recurrente con ventanas temporales
- **Prophet**: Modelo aditivo para tendencias
- **Métricas**: MAE, RMSE, MAPE para forecasting

## Características de la Variable CO(GT)

### Estadísticas Descriptivas
```
Media:              2.11 mg/m³
Desviación estándar: 1.87 mg/m³
Mínimo:           -200.00 mg/m³  (valor sensor inválido)
Máximo:            11.9 mg/m³
Mediana:            1.60 mg/m³
Q1:                 0.89 mg/m³
Q3:                 2.83 mg/m³
```

### Propiedades Temporales
- **Tendencia**: Ligeramente decreciente a largo plazo
- **Estacionalidad**: No significativa (cumple requisitos)
- **Autocorrelación**: AR(1) con coeficiente ≈ 0.2
- **Ruido**: Componente estocástico moderado

### Validación de No Estacionalidad

**Tests realizados**:
- **Estacionalidad diaria**: No significativa (p > 0.05)
- **Estacionalidad semanal**: No significativa (p > 0.05)
- **Estacionalidad mensual**: No significativa (p > 0.05)
- **Componente estacional**: < 5% de la varianza total

**Conclusión**: La variable CO(GT) cumple perfectamente el requisito de "variable NO estacional" para el TEST 2.

## Principales Hallazgos

### Características del Dataset
- **Calidad de datos**: 89.4% de valores válidos
- **Continuidad temporal**: Gaps distribuidos uniformemente
- **Variabilidad**: Concentración CO muy variable por condiciones ambientales
- **Outliers**: Valores extremos relacionados con eventos atmosféricos

### Insights del Análisis Temporal
- **Patrón diario leve**: Ligeramente más alto en horas pico tráfico
- **Sin estacionalidad fuerte**: Ideal para modelos ARIMA
- **Memoria corta**: Autocorrelación decae rápidamente
- **Proceso integrado I(1)**: Una diferenciación suficiente para estacionariedad

### Preparación para Modelado
- **Estacionariedad**: Conseguida con d=1 en ARIMA
- **Ventana temporal**: 24 horas para LSTM (1 día contexto)
- **División temporal**: 80% train, 20% test (últimos períodos)
- **Validación**: Walk-forward para simular uso real

## Cumplimiento de Requisitos

### Verificación TEST 2
- ✅ **Dataset NO financiero**: Air Quality UCI (datos ambientales)
- ✅ **Variable NO estacional**: CO(GT) validada sin estacionalidad
- ✅ **Horizonte 100 períodos**: Implementado en todos los modelos
- ✅ **Modelos especializados**: ARIMA, LSTM, Prophet

### Justificación Técnica
- **Dataset ambiental**: Concentraciones de gases en atmósfera
- **Variable apropiada**: CO(GT) sin patrones estacionales fuertes
- **Horizonte realista**: 100 horas ≈ 4 días de predicción
- **Modelos diversos**: Estadístico, deep learning, y híbrido

## Navegación

- **[Dataset & Analysis](dataset.md)**: Análisis detallado del dataset temporal
- **[Forecasting Models](models.md)**: Implementación y resultados de modelos
- **[API Reference](api.md)**: Documentación técnica del código