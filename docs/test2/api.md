# API Reference - TEST 2

Esta sección documenta las clases y funciones principales del proyecto de forecasting de series temporales.

## Módulo de Datos

::: ttac_test_ds_timeseries.data.load_gas_sensor

## Módulos de Modelos

### ARIMA Model

::: ttac_test_ds_timeseries.models.arima_model

### LSTM Model

::: ttac_test_ds_timeseries.models.lstm_model

### Prophet Model

::: ttac_test_ds_timeseries.models.prophet_model

## Scripts de Línea de Comandos

### train_model.py

Script para entrenar modelos de forecasting:

```bash
python train_model.py [--model MODEL_TYPE] [--horizon PERIODS] [--output OUTPUT_PATH]
```

**Parámetros:**
- `--model`: Tipo de modelo ('arima', 'lstm', 'prophet'). Default: 'arima'
- `--horizon`: Número de períodos a predecir. Default: 100
- `--output`: Ruta para guardar el modelo entrenado

**Ejemplo:**
```bash
python train_model.py --model arima --horizon 100 --output models/co_arima_model.pkl
```

### forecast.py

Script para realizar predicciones temporales:

```bash
python forecast.py --model MODEL_PATH --horizon PERIODS [--confidence LEVEL] [--output OUTPUT_PATH]
```

**Parámetros:**
- `--model`: Ruta del modelo entrenado
- `--horizon`: Número de períodos futuros a predecir
- `--confidence`: Nivel de confianza para intervalos (0.95 default)
- `--output`: Ruta para guardar las predicciones (opcional)

**Ejemplo:**
```bash
python forecast.py --model models/co_arima_model.pkl --horizon 100 --confidence 0.95
```

### process.py

Script para preprocesamiento de datos temporales:

```bash
python process.py --input DATA_PATH [--interpolate] [--max_gap HOURS] [--output OUTPUT_PATH]
```

**Parámetros:**
- `--input`: Ruta del archivo CSV con datos temporales
- `--interpolate`: Activar interpolación de valores faltantes
- `--max_gap`: Máximo gap en horas para interpolación (default: 6)
- `--output`: Ruta para guardar datos procesados

**Ejemplo:**
```bash
python process.py --input data/raw/air_quality.csv --interpolate --max_gap 6
```

## Utilidades de Series Temporales

### Funciones de Análisis

```python
def check_stationarity(series, significance_level=0.05):
    """
    Realiza test de estacionariedad Augmented Dickey-Fuller.
    
    Args:
        series: Serie temporal a analizar
        significance_level: Nivel de significancia del test
        
    Returns:
        dict: Resultados del test con estadístico y p-value
    """
```

```python
def analyze_seasonality(series, periods=[24, 168, 8760]):
    """
    Analiza componentes estacionales en series temporales.
    
    Args:
        series: Serie temporal
        periods: Lista de períodos a analizar (diario, semanal, anual)
        
    Returns:
        dict: Resultados de tests de estacionalidad
    """
```

```python
def decompose_timeseries(series, model='additive', period=24):
    """
    Descompone serie temporal en tendencia, estacionalidad y residuo.
    
    Args:
        series: Serie temporal
        model: Tipo de descomposición ('additive' o 'multiplicative')
        period: Período estacional
        
    Returns:
        statsmodels.tsa.seasonal.DecomposeResult: Componentes descompuestos
    """
```

### Funciones de Evaluación

```python
def forecast_metrics(actual, predicted):
    """
    Calcula métricas estándar para evaluación de forecasting.
    
    Args:
        actual: Valores reales
        predicted: Valores predichos
        
    Returns:
        dict: Métricas MAE, RMSE, MAPE, R²
    """
```

```python
def plot_forecast_results(actual, predicted, confidence_intervals=None):
    """
    Visualiza resultados de forecasting con intervalos de confianza.
    
    Args:
        actual: Serie temporal real
        predicted: Predicciones del modelo
        confidence_intervals: Intervalos de confianza (opcional)
        
    Returns:
        matplotlib.figure.Figure: Gráfico de forecasting
    """
```

### Funciones de Validación

```python
def walk_forward_validation(model, data, train_size, horizon, step=1):
    """
    Realiza validación walk-forward para series temporales.
    
    Args:
        model: Modelo de forecasting
        data: Serie temporal completa
        train_size: Tamaño de ventana de entrenamiento
        horizon: Horizonte de predicción
        step: Paso de la ventana deslizante
        
    Returns:
        list: Predicciones y valores reales para cada ventana
    """
```

## Configuración y Constantes

### Parámetros por Defecto

```python
# Configuración ARIMA
DEFAULT_ARIMA_PARAMS = {
    'order': (1, 1, 1),
    'enforce_stationarity': True,
    'enforce_invertibility': True,
    'trend': None
}

# Configuración LSTM
DEFAULT_LSTM_PARAMS = {
    'look_back': 24,
    'units': [50, 50],
    'dropout': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2
}

# Configuración Prophet
DEFAULT_PROPHET_PARAMS = {
    'growth': 'linear',
    'seasonality_mode': 'additive',
    'yearly_seasonality': False,
    'weekly_seasonality': False,
    'daily_seasonality': True,
    'interval_width': 0.95
}
```

### Constantes de Proyecto

```python
# Horizonte objetivo TEST 2
FORECAST_HORIZON = 100

# Frecuencia de datos
DATA_FREQUENCY = 'H'  # Horarios

# Variable objetivo
TARGET_VARIABLE = 'CO(GT)'

# Métricas de evaluación
EVALUATION_METRICS = ['MAE', 'RMSE', 'MAPE', 'R2']

# Nivel de confianza por defecto
DEFAULT_CONFIDENCE_LEVEL = 0.95
```

## Ejemplos de Uso

### Entrenamiento ARIMA Básico

```python
from ttac_test_ds_timeseries.data.load_gas_sensor import load_air_quality_data
from ttac_test_ds_timeseries.models.arima_model import AirQualityARIMA

# Cargar datos
co_data = load_air_quality_data()

# Entrenar modelo ARIMA
arima_model = AirQualityARIMA(order=(1, 1, 1))
arima_model.fit(co_data)

# Realizar forecasting
forecast = arima_model.forecast(horizon=100)
confidence_intervals = arima_model.forecast_intervals(horizon=100, alpha=0.05)

print(f"Forecast primeros 10 valores: {forecast[:10]}")
```

### Pipeline Completo de Forecasting

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ttac_test_ds_timeseries.models.arima_model import AirQualityARIMA

# 1. Cargar y preprocesar datos
co_data = load_air_quality_data()
co_clean = preprocess_timeseries(co_data, interpolate=True, max_gap=6)

# 2. División temporal
train_size = int(0.8 * len(co_clean))
train_data = co_clean[:train_size]
test_data = co_clean[train_size:]

# 3. Entrenar modelo
model = AirQualityARIMA()
model.fit(train_data)

# 4. Predecir últimos 100 períodos
forecast_100 = model.forecast(horizon=100)
actual_100 = test_data[-100:]

# 5. Evaluar
mae = mean_absolute_error(actual_100, forecast_100)
mape = np.mean(np.abs((actual_100 - forecast_100) / actual_100)) * 100

print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# 6. Guardar modelo
model.save('models/arima_final.pkl')
```

### Comparación de Modelos

```python
from ttac_test_ds_timeseries.models import AirQualityARIMA, AirQualityLSTM, AirQualityProphet

# Configurar modelos
models = {
    'ARIMA': AirQualityARIMA(order=(1, 1, 1)),
    'LSTM': AirQualityLSTM(look_back=24, units=[50, 50]),
    'Prophet': AirQualityProphet(daily_seasonality=True)
}

# Entrenar y evaluar cada modelo
results = {}
for name, model in models.items():
    # Entrenar
    model.fit(train_data)
    
    # Predecir
    forecast = model.forecast(horizon=100)
    
    # Evaluar
    metrics = forecast_metrics(actual_100, forecast)
    results[name] = metrics

# Mostrar resultados
for name, metrics in results.items():
    print(f"{name}: MAPE = {metrics['MAPE']:.2f}%")
```

### Validación Walk-Forward

```python
def comprehensive_validation(model_class, data, **model_params):
    """
    Validación completa con múltiples ventanas temporales.
    """
    # Configurar validación
    window_sizes = [500, 1000, 1500]
    horizons = [50, 100, 150]
    
    results = []
    
    for window_size in window_sizes:
        for horizon in horizons:
            # Validación walk-forward
            predictions, actuals = walk_forward_validation(
                model_class(**model_params),
                data,
                train_size=window_size,
                horizon=horizon
            )
            
            # Calcular métricas promedio
            avg_metrics = {}
            for pred, actual in zip(predictions, actuals):
                metrics = forecast_metrics(actual, pred)
                for key, value in metrics.items():
                    avg_metrics.setdefault(key, []).append(value)
            
            # Promediar métricas
            for key in avg_metrics:
                avg_metrics[key] = np.mean(avg_metrics[key])
            
            results.append({
                'window_size': window_size,
                'horizon': horizon,
                **avg_metrics
            })
    
    return pd.DataFrame(results)

# Ejecutar validación completa
validation_results = comprehensive_validation(
    AirQualityARIMA, 
    co_clean, 
    order=(1, 1, 1)
)

print(validation_results)
```

## Testing

### Ejecución de Tests

```bash
# Ejecutar todos los tests
pytest tests/

# Tests específicos
pytest tests/test_data.py
pytest tests/test_models.py

# Con coverage detallado
pytest tests/ --cov=src/ttac_test_ds_timeseries --cov-report=html
```

### Estructura de Tests

```python
# test_data.py
def test_load_air_quality_data():
    """Test carga correcta del dataset UCI."""
    
def test_preprocess_timeseries():
    """Test preprocesamiento de series temporales."""
    
def test_stationarity_check():
    """Test validación de estacionariedad."""

# test_models.py
def test_arima_fitting():
    """Test entrenamiento del modelo ARIMA."""
    
def test_arima_forecasting():
    """Test predicción con ARIMA."""
    
def test_lstm_architecture():
    """Test construcción de arquitectura LSTM."""
    
def test_prophet_configuration():
    """Test configuración del modelo Prophet."""

def test_forecast_metrics():
    """Test cálculo de métricas de evaluación."""
```

### Tests de Compliance TEST 2

```python
def test_forecast_horizon_compliance():
    """Verifica que el modelo predice exactamente 100 períodos."""
    model = AirQualityARIMA()
    model.fit(train_data)
    forecast = model.forecast(horizon=100)
    
    assert len(forecast) == 100, "Debe predecir exactamente 100 períodos"

def test_non_seasonal_variable_compliance():
    """Verifica que CO(GT) no tiene estacionalidad significativa."""
    seasonality_results = analyze_seasonality(co_data)
    
    for period, p_value in seasonality_results.items():
        assert p_value > 0.05, f"Variable no debe ser estacional (período {period})"

def test_non_financial_dataset_compliance():
    """Verifica que el dataset es ambiental, no financiero."""
    metadata = get_dataset_metadata()
    
    assert metadata['domain'] == 'environmental', "Dataset debe ser ambiental"
    assert metadata['type'] != 'financial', "Dataset no debe ser financiero"
```