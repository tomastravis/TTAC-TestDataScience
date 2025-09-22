# API Reference - TEST 1

Esta sección documenta las clases y funciones principales del proyecto de clasificación de calidad de vinos.

## Módulo de Datos

::: ttac_test_ds_classification.data

## Módulo de Modelos

::: ttac_test_ds_classification.models

## Scripts de Línea de Comandos

### train_model.py

Script para entrenar modelos de clasificación:

```bash
python train_model.py [--model MODEL_TYPE] [--output OUTPUT_PATH]
```

**Parámetros:**
- `--model`: Tipo de modelo ('rf', 'svm', 'xgboost'). Default: 'rf'
- `--output`: Ruta para guardar el modelo entrenado

**Ejemplo:**
```bash
python train_model.py --model rf --output models/wine_rf_model.pkl
```

### inference.py

Script para realizar predicciones:

```bash
python inference.py --model MODEL_PATH --data DATA_PATH [--output OUTPUT_PATH]
```

**Parámetros:**
- `--model`: Ruta del modelo entrenado
- `--data`: Ruta del archivo CSV con datos para predecir
- `--output`: Ruta para guardar las predicciones (opcional)

**Ejemplo:**
```bash
python inference.py --model models/wine_rf_model.pkl --data data/new_wines.csv
```

## Utilidades y Funciones de Apoyo

### Funciones de Evaluación

```python
def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo con métricas estándar de clasificación.
    
    Args:
        model: Modelo entrenado de scikit-learn
        X_test: Features de test
        y_test: Labels verdaderas de test
        
    Returns:
        dict: Diccionario con métricas de evaluación
    """
```

### Funciones de Visualización

```python
def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Genera matriz de confusión visualizada.
    
    Args:
        y_true: Labels verdaderas
        y_pred: Predicciones del modelo
        classes: Lista de nombres de clases
        
    Returns:
        matplotlib.figure.Figure: Figura de la matriz
    """
```

### Funciones de Feature Importance

```python
def plot_feature_importance(model, feature_names, top_k=10):
    """
    Visualiza la importancia de features para modelos tree-based.
    
    Args:
        model: Modelo entrenado (RandomForest, XGBoost)
        feature_names: Lista de nombres de features
        top_k: Número de features más importantes a mostrar
        
    Returns:
        matplotlib.figure.Figure: Gráfico de importancia
    """
```

## Configuración y Constantes

### Parámetros por Defecto

```python
# Configuración de modelos
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

DEFAULT_SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'random_state': 42,
    'probability': True
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42
}
```

### Rutas de Archivos

```python
# Estructura de directorios
DATA_RAW = 'data/raw/'
DATA_PROCESSED = 'data/processed/'
DATA_FINAL = 'data/final/'
MODELS_DIR = 'models/'
NOTEBOOKS_DIR = 'notebooks/'
```

## Ejemplos de Uso

### Entrenamiento Básico

```python
from ttac_test_ds_classification.data import load_wine_data
from ttac_test_ds_classification.models import WineQualityClassifier

# Cargar datos
X, y = load_wine_data()

# Entrenar modelo
classifier = WineQualityClassifier(model_type='rf')
classifier.fit(X, y)

# Evaluar
results = classifier.evaluate()
print(f"Accuracy: {results['accuracy']:.3f}")
```

### Predicción en Nuevos Datos

```python
import pandas as pd
from ttac_test_ds_classification.models import WineQualityClassifier

# Cargar modelo entrenado
classifier = WineQualityClassifier.load('models/wine_rf_model.pkl')

# Cargar nuevos datos
new_wines = pd.read_csv('data/new_wines.csv')

# Realizar predicciones
predictions = classifier.predict(new_wines)
probabilities = classifier.predict_proba(new_wines)

print(f"Predicciones: {predictions}")
print(f"Probabilidades: {probabilities}")
```

### Pipeline Completo

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ttac_test_ds_classification.data import load_wine_data
from ttac_test_ds_classification.models import WineQualityClassifier

# 1. Cargar y dividir datos
X, y = load_wine_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Preprocesar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenar
classifier = WineQualityClassifier(model_type='rf')
classifier.fit(X_train_scaled, y_train)

# 4. Evaluar
y_pred = classifier.predict(X_test_scaled)
results = classifier.evaluate(X_test_scaled, y_test)

# 5. Guardar modelo
classifier.save('models/wine_classifier_final.pkl')
```

## Testing

### Ejecución de Tests

```bash
# Ejecutar todos los tests
pytest tests/

# Ejecutar tests específicos
pytest tests/test_train_model.py
pytest tests/test_process.py

# Con coverage
pytest tests/ --cov=src/
```

### Estructura de Tests

```python
# test_train_model.py
def test_model_training():
    """Test que el modelo se entrena correctamente."""
    
def test_model_evaluation():
    """Test que la evaluación produce métricas válidas."""
    
def test_model_persistence():
    """Test que el modelo se guarda y carga correctamente."""

# test_process.py  
def test_data_loading():
    """Test que los datos se cargan correctamente."""
    
def test_data_preprocessing():
    """Test que el preprocesamiento funciona."""
    
def test_train_test_split():
    """Test que la división train/test es estratificada."""
```