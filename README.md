# TTAC TestDataScience - Proyectos de Machine Learning

🎯 **Dos proyectos completos de Data Science con datasets reales UCI ML Repository**

Este repositorio contiene dos proyectos end-to-end de Machine Learning ejecutables desde notebooks interactivos:

## **TEST 1 - Clasificación** 
**Wine Quality Classification** con Random Forest, SVM y XGBoost  
Dataset: UCI Wine Quality (6,497 muestras) → **69.31% Accuracy**

## **TEST 2 - Series Temporales**
**Gas Sensor Array Drift Forecasting** con Random Forest, XGBoost y ARIMA  
Dataset: UCI Gas Sensor Drift (144 observaciones) → **2.58% MAPE**

---

## Configuración Rápida

### 1. Crear Entorno Virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows
```

### 2. Instalar Dependencias
```bash
# Para ambos proyectos + documentación
pip install -r requirements-docs.txt
cd TTAC-TestDataScience-1 && pip install -r requirements.txt && cd ..
cd TTAC-TestDataScience-2 && pip install -r requirements.txt && cd ..
```

### 3. Ejecutar el Core del Proyecto - Notebooks
```bash
# Iniciar Jupyter para acceder a los notebooks principales
jupyter notebook

# CORE del proyecto - Ejecutar estos notebooks en orden:

# TEST 1 - Clasificación Wine Quality:
# 1. TTAC-TestDataScience-1/notebooks/01_eda_classification.ipynb
# 2. TTAC-TestDataScience-1/notebooks/02_modeling_classification.ipynb

# TEST 2 - Series Temporales Gas Sensor:
# 1. TTAC-TestDataScience-2/notebooks/01_eda_timeseries.ipynb  
# 2. TTAC-TestDataScience-2/notebooks/02_modeling_forecasting.ipynb
```

### 4. Ver Resultados HTML (Opcional)
```bash
# Los notebooks también están disponibles como HTML embebido:
# - TTAC-TestDataScience-1/notebooks/01_eda_classification.html
# - TTAC-TestDataScience-1/notebooks/02_modeling_classification.html  
# - TTAC-TestDataScience-2/notebooks/01_eda_timeseries.html
# - TTAC-TestDataScience-2/notebooks/02_modeling_forecasting.html

# Documentación completa (opcional):
mkdocs serve  # Abre http://127.0.0.1:8000
```

---

## Estructura del Proyecto

```
TTAC-TestDataScience/
├── docs/                         # Documentación MkDocs completa
├── TTAC-TestDataScience-1/       # TEST 1: Wine Quality Classification
│   ├── notebooks/
│   │   ├── 01_eda_classification.ipynb     # 🎯 CORE - EDA Análisis
│   │   ├── 01_eda_classification.html      # HTML embebido
│   │   ├── 02_modeling_classification.ipynb # 🎯 CORE - Modelado
│   │   └── 02_modeling_classification.html # HTML embebido
│   └── ...
├── TTAC-TestDataScience-2/       # TEST 2: Gas Sensor Forecasting  
│   ├── notebooks/
│   │   ├── 01_eda_timeseries.ipynb         # 🎯 CORE - EDA Temporal
│   │   ├── 01_eda_timeseries.html          # HTML embebido
│   │   ├── 02_modeling_forecasting.ipynb   # 🎯 CORE - Forecasting
│   │   └── 02_modeling_forecasting.html    # HTML embebido
│   └── ...
├── mkdocs.yml                    # Configuración documentación
└── requirements-docs.txt         # Dependencias MkDocs
```

## 🎯 Notebooks Principales (CORE del Proyecto)

Los **4 notebooks principales** que demuestran todo el flujo de trabajo:

1. **TEST 1 - Clasificación**: `TTAC-TestDataScience-1/notebooks/`
   - `01_eda_classification.ipynb` - EDA completo Wine Quality
   - `02_modeling_classification.ipynb` - Random Forest, SVM, XGBoost

2. **TEST 2 - Series Temporales**: `TTAC-TestDataScience-2/notebooks/`  
   - `01_eda_timeseries.ipynb` - EDA Gas Sensor Array Drift
   - `02_modeling_forecasting.ipynb` - Random Forest, XGBoost, ARIMA

Cada notebook también está disponible como **HTML embebido** para visualización sin Jupyter.

---

## 🚀 Inicio Rápido - Solo 2 Pasos

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements-docs.txt
   cd TTAC-TestDataScience-1 && pip install -r requirements.txt && cd ..
   cd TTAC-TestDataScience-2 && pip install -r requirements.txt && cd ..
   ```

2. **Ejecutar notebooks**:
   ```bash
   jupyter notebook
   # Navegar a las carpetas notebooks/ y ejecutar los .ipynb
   ```

## 📚 Documentación Opcional

Para documentación técnica completa (opcional):
```bash
mkdocs serve  # http://127.0.0.1:8000
```