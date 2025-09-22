# Setup & Usage

## Requisitos del Sistema

### Dependencias Principales
- **Python**: 3.12 o superior
- **Git**: Para clonar el repositorio
- **Virtual Environment**: Recomendado para aislamiento

### Librerías Python Requeridas
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
statsmodels>=0.14.0
tensorflow>=2.13.0
prophet>=1.1.0
pytest>=7.0.0
```

## Instalación Rápida

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tomastravis/TTAC-TestDataScience.git
cd TTAC-TestDataScience
```

### 2. Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno (macOS/Linux)
source .venv/bin/activate

# Activar entorno (Windows)
.venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
# Instalar dependencias del TEST 1
cd TTAC-TestDataScience-1
pip install -r requirements.txt

# Instalar dependencias del TEST 2
cd ../TTAC-TestDataScience-2
pip install -r requirements.txt
```

### 4. Verificar Instalación
```bash
# Ejecutar script de verificación
python verify_setup.py
```

## Uso de los Proyectos

### TEST 1: Wine Quality Classification

#### Análisis Exploratorio
```bash
cd TTAC-TestDataScience-1

# Iniciar Jupyter para EDA
jupyter notebook notebooks/01_eda_classification.ipynb
```

#### Entrenamiento de Modelos
```bash
# Entrenar Random Forest (recomendado)
python src/train_model.py --model rf --output models/wine_rf_model.pkl

# Entrenar SVM
python src/train_model.py --model svm --output models/wine_svm_model.pkl

# Entrenar XGBoost
python src/train_model.py --model xgboost --output models/wine_xgb_model.pkl
```

#### Realizar Predicciones
```bash
# Predicciones con modelo entrenado
python src/inference.py \
    --model models/wine_rf_model.pkl \
    --data data/new_wines.csv \
    --output predictions/wine_predictions.csv
```

#### Ejecutar Tests
```bash
# Tests completos
pytest tests/ -v

# Test específico
pytest tests/test_train_model.py -v
```

### TEST 2: Air Quality Forecasting

#### Análisis Temporal
```bash
cd TTAC-TestDataScience-2

# Iniciar Jupyter para análisis temporal
jupyter notebook notebooks/01_eda_timeseries.ipynb
```

#### Entrenamiento de Modelos
```bash
# Entrenar ARIMA (recomendado)
python src/train_model.py --model arima --horizon 100 --output models/co_arima_model.pkl

# Entrenar LSTM
python src/train_model.py --model lstm --horizon 100 --output models/co_lstm_model.pkl

# Entrenar Prophet
python src/train_model.py --model prophet --horizon 100 --output models/co_prophet_model.pkl
```

#### Realizar Forecasting
```bash
# Predicción de 100 períodos
python src/forecast.py \
    --model models/co_arima_model.pkl \
    --horizon 100 \
    --confidence 0.95 \
    --output forecasts/co_forecast_100h.csv
```

#### Ejecutar Tests
```bash
# Tests completos
pytest tests/ -v

# Test de compliance TEST 2
pytest tests/test_models.py::test_forecast_horizon_compliance -v
```

## Estructura de Archivos

### Organización General
```
TTAC-TestDataScience/
├── README.md                    # Documentación principal
├── .gitignore                   # Configuración Git
├── mkdocs.yml                   # Configuración documentación
├── docs/                        # Documentación MkDocs
│   ├── index.md
│   ├── test1/                   # Docs TEST 1
│   ├── test2/                   # Docs TEST 2
│   └── setup.md                 # Esta página
├── TTAC-TestDataScience-1/      # Proyecto clasificación
│   ├── src/
│   ├── notebooks/
│   ├── tests/
│   ├── data/
│   ├── models/
│   └── requirements.txt
└── TTAC-TestDataScience-2/      # Proyecto series temporales
    ├── src/
    ├── notebooks/
    ├── tests/
    ├── data/
    ├── models/
    └── requirements.txt
```

### Datos de Entrada

#### TEST 1: Wine Quality
```
data/raw/wine_quality_real.csv          # Dataset UCI original
data/processed/wine_quality_clean.csv   # Datos preprocesados
data/final/wine_quality_final.csv       # Datos finales para modelado
```

#### TEST 2: Air Quality
```
data/raw/air_quality_uci.csv           # Dataset UCI original
data/processed/co_gt_clean.csv          # Serie temporal limpia
data/final/co_gt_modeling.csv          # Datos finales para forecasting
```

## Configuración de Desarrollo

### Variables de Entorno
```bash
# Configurar Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/TTAC-TestDataScience-1/src"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/TTAC-TestDataScience-2/src"

# Configurar Jupyter
export JUPYTER_CONFIG_DIR="$(pwd)/.jupyter"
```

### Configuración IDE
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true
}
```

## Resolución de Problemas

### Problemas Comunes

#### Error: "Module not found"
```bash
# Verificar instalación
pip list | grep -E "(pandas|scikit-learn|tensorflow)"

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

#### Error: "CUDA not available" (LSTM)
```bash
# Instalar CPU-only TensorFlow
pip uninstall tensorflow
pip install tensorflow-cpu
```

#### Error: "Jupyter kernel not found"
```bash
# Instalar kernel en entorno virtual
python -m ipykernel install --user --name=.venv --display-name="TTAC DataScience"
```

### Verificación de Setup

#### Script de Verificación Completa
```python
# verify_setup.py
import sys
import subprocess
import importlib

def check_python_version():
    """Verifica versión de Python."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 12:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requiere 3.12+)")
        return False

def check_packages():
    """Verifica paquetes requeridos."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'statsmodels', 'tensorflow', 'prophet'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def check_data_files():
    """Verifica archivos de datos."""
    import os
    
    data_files = [
        'TTAC-TestDataScience-1/data/raw/wine_quality_real.csv',
        'TTAC-TestDataScience-2/data/raw/'  # Directorio debe existir
    ]
    
    all_exist = True
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("🔍 Verificando configuración TTAC Data Science Tests\n")
    
    checks = [
        ("Versión Python", check_python_version()),
        ("Paquetes Python", check_packages()),
        ("Archivos de datos", check_data_files())
    ]
    
    all_passed = all(result for _, result in checks)
    
    print(f"\n{'✅ Setup completo' if all_passed else '❌ Setup incompleto'}")
    
    if not all_passed:
        print("\n📝 Para resolver problemas:")
        print("1. Verificar versión Python >= 3.12")
        print("2. Ejecutar: pip install -r requirements.txt")
        print("3. Verificar que los datasets UCI están descargados")
```

## Desarrollo y Contribución

### Flujo de Trabajo
1. **Fork del repositorio**
2. **Crear rama feature**: `git checkout -b feature/nueva-funcionalidad`
3. **Desarrollar y testear**: `pytest tests/`
4. **Commit y push**: `git commit -m "Descripción clara"`
5. **Crear Pull Request**

### Estándares de Código
- **Type hints**: Usar tipado Python
- **Docstrings**: Documentar todas las funciones
- **Tests**: Cobertura mínima 80%
- **Linting**: Seguir PEP 8

### Tests Obligatorios
```bash
# Antes de commit
pytest tests/ --cov=src --cov-report=term-missing
mypy src/
```

## Deployment y Producción

### Docker (Opcional)
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "src/inference.py"]
```

### GitHub Actions (CI/CD)
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src
```

## Soporte y Documentación

### Recursos Adicionales
- Documentación en local: ejecutar `mkdocs serve` desde la raíz y abrir http://127.0.0.1:8000
- Notebooks interactivos: Disponibles en `notebooks/`
- API Reference: Documentación automática con MkDocstrings
- Issues: https://github.com/tomastravis/TTAC-TestDataScience/issues

### Contacto
- **Autor**: Tomás Travis Alonso Cremnitz
- **Email**: [Disponible en perfil GitHub]
- **Portfolio**: Machine Learning Engineer
- **Especialización**: Classification & Time Series Forecasting