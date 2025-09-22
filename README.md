# TTAC TestDataScience - Proyectos de Machine Learning

Este repositorio contiene dos proyectos completos de Data Science:

## **TEST 1 - Clasificación** 
**Wine Quality Classification** con Random Forest, SVM y XGBoost  
Dataset: UCI Wine Quality (6,497 muestras) → **69.31% Accuracy**

## **TEST 2 - Series Temporales**
**Air Quality Forecasting** con ARIMA, LSTM y Prophet  
Dataset: UCI Air Quality (9,358 registros horarios) → **21.84% MAPE**

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

### 3. Usar el Proyecto
```bash
# Ver documentación completa
mkdocs serve  # → http://127.0.0.1:8000

# O ejecutar notebooks directamente
jupyter notebook
```

**📖 Documentación completa**: https://tomastravis.github.io/TTAC-TestDataScience/

---

## Estructura del Proyecto

```
TTAC-TestDataScience/
├── docs/                         # Documentación MkDocs completa
├── TTAC-TestDataScience-1/       # TEST 1: Wine Quality Classification
├── TTAC-TestDataScience-2/       # TEST 2: Air Quality Forecasting
├── mkdocs.yml                    # Configuración documentación
└── requirements-docs.txt         # Dependencias MkDocs
```

**Para información detallada, instalación paso a paso y uso avanzado consulta la [documentación completa](https://tomastravis.github.io/TTAC-TestDataScience/).**