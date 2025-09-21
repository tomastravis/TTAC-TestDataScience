# TTAC TestDataScience-1: Clasificación de Calidad de Vinos

## 📊 Información del Dataset

**Fuente**: UCI Machine Learning Repository - Wine Quality Dataset

**Citación**: 
```
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. 
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
```

**Descripción**: 
- **Dataset combinado**: 6,497 muestras totales
- **Características**: 11 propiedades fisicoquímicas (alcohol, acidez, etc.)
- **Variable objetivo**: calidad del vino (puntuación 3-9)
- **Tarea**: Problema de clasificación multiclase

## 🚀 Inicio Rápido

### Opción 1: Notebooks (Recomendado)
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Iniciar Jupyter
jupyter notebook

# 3. Ejecutar notebooks en orden:
#    - notebooks/01_eda_classification.ipynb
#    - notebooks/02_modeling_classification.ipynb
```

### Opción 2: Scripts CLI
```bash
# Entrenar modelo
python src/train_model.py --model=random_forest

# Hacer predicción
python src/inference.py --model=models/random_forest_model.pkl
```

## 📁 Estructura del Proyecto

```bash
.
├── data/raw/                      # Datos UCI Wine Quality
├── models/                        # Modelos entrenados (.pkl)
├── notebooks/                     # Notebooks de Jupyter
├── src/                          # Código fuente
│   ├── train_model.py            # Script de entrenamiento
│   ├── inference.py              # Script de predicción
│   └── ttac_test_ds_classification/  # Paquete principal
├── tests/                        # Tests unitarios
└── requirements.txt              # Dependencias de Python
```

## 🤖 Modelos de Machine Learning

- **Random Forest**: Modelo principal (precisión: ~69%)
- **SVM**: Modelo alternativo 
- **XGBoost**: Modelo de boosting

## 🧪 Pruebas

```bash
python -m pytest tests/
```

---

**Autor**: Tomás Travis Alonso Cremnitz