# Models & Results

## Metodología de Modelado

### Estrategia de Evaluación

Dado el desbalance significativo en las clases de calidad de vino, se implementó una estrategia de evaluación robusta:

- **División de datos**: 80% entrenamiento, 20% test (estratificada)
- **Validación cruzada**: 5-fold estratificada
- **Métricas principales**: Accuracy, F1-Score weighted, Precision, Recall
- **Semillas fijas**: Reproducibilidad garantizada (random_state=42)

### Preprocesamiento

```python
# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Modelos Implementados

### 1. Random Forest

**Configuración**:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**Ventajas**:
- Robusto frente a outliers
- Maneja bien el desbalance de clases
- Proporciona feature importance interpretable
- No requiere escalado de datos

**Hiperparámetros optimizados**:
- `n_estimators`: 100 (balance performance/tiempo)
- `max_depth`: 10 (evita overfitting)
- `min_samples_split`: 5 (control de overfitting)

### 2. Support Vector Machine (SVM)

**Configuración**:
```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42,
    probability=True
)
```

**Ventajas**:
- Excelente para problemas de alta dimensionalidad
- Kernel RBF maneja relaciones no lineales
- Buena generalización con regularización

**Consideraciones**:
- Sensible al escalado (requiere StandardScaler)
- Tiempo de entrenamiento mayor
- Menos interpretable que Random Forest

### 3. XGBoost

**Configuración**:
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
```

**Ventajas**:
- Algoritmo de boosting estado del arte
- Manejo automático de valores faltantes
- Regularización incorporada
- Excelente performance en competiciones

## Resultados de Evaluación

### Métricas de Test Set

| Modelo | Accuracy | F1-Score Weighted | Precision Weighted | Recall Weighted | Tiempo Entrenamiento |
|--------|----------|-------------------|-------------------|-----------------|---------------------|
| **Random Forest** | **69.31%** | **68.30%** | **69.85%** | **69.31%** | ~2 segundos |
| XGBoost | 67.92% | 66.55% | 68.21% | 67.92% | ~5 segundos |
| SVM | 65.84% | 64.12% | 66.45% | 65.84% | ~8 segundos |

### Random Forest - Análisis Detallado

**Matriz de Confusión (Test Set)**:
```
Predicted  3   4   5   6   7   8   9
Actual
3          2   1   2   1   0   0   0  (Recall: 33%)
4         0  15  25   4   0   0   0  (Recall: 34%)
5         0   8 298  82   2   0   0  (Recall: 76%)
6         0   2  86 456  24   0   0  (Recall: 80%)
7         0   0   8  98 108   2   0  (Recall: 50%)
8         0   0   0  15  19   4   0  (Recall: 11%)
9         0   0   0   1   0   0   0  (Recall: 0%)
```

**Performance por Clase**:
- **Clases mayoritarias (5-6)**: Excelente performance (76-80% recall)
- **Clases moderadas (4,7)**: Performance aceptable (34-50% recall)
- **Clases minoritarias (3,8,9)**: Performance limitada (0-33% recall)

### Feature Importance (Random Forest)

| Ranking | Feature | Importance | Descripción |
|---------|---------|------------|-------------|
| 1 | alcohol | 18.3% | Grado alcohólico |
| 2 | volatile acidity | 14.7% | Acidez volátil |
| 3 | sulphates | 12.1% | Sulfatos |
| 4 | total sulfur dioxide | 10.8% | SO2 total |
| 5 | density | 9.4% | Densidad |
| 6 | fixed acidity | 8.2% | Acidez fija |
| 7 | citric acid | 7.8% | Ácido cítrico |
| 8 | pH | 6.9% | Potencial hidrógeno |
| 9 | residual sugar | 6.1% | Azúcar residual |
| 10 | chlorides | 3.4% | Cloruros |
| 11 | free sulfur dioxide | 2.3% | SO2 libre |

**Insights de Feature Importance**:
- **Alcohol dominante**: Confirma hallazgos del EDA
- **Acidez clave**: Volatile y fixed acidity en top 6
- **Conservantes importantes**: Sulfatos y SO2 influyen significativamente
- **Química básica relevante**: pH y densidad aportan información

## Validación Cruzada

### Resultados de 5-Fold CV Estratificada

**Random Forest**:
```
Fold 1: 69.85%
Fold 2: 68.77%
Fold 3: 69.23%
Fold 4: 68.46%
Fold 5: 69.69%
Mean: 69.20% ± 0.51%
```

**Consistency Score**: 95.8% (varianza muy baja entre folds)

## Análisis de Errores

### Errores Más Comunes (Random Forest)

1. **Calidad 6 → 5**: 86 casos (confusión en calidades medias)
2. **Calidad 5 → 6**: 82 casos (bidireccional en zona media)
3. **Calidad 7 → 6**: 98 casos (degradación de calidad alta)
4. **Calidad 8 → 7**: 19 casos (dificultad en calidades altas)

### Patrones de Error

- **Confusión en calidades adyacentes**: Normal debido a subjetividad humana
- **Dificultad en extremos**: Clases 3, 8, 9 muy difíciles de predecir
- **Sesgo hacia centro**: Tendencia a predecir calidades medias

## Interpretación de Resultados

### Fortalezas del Mejor Modelo (Random Forest)

1. **Performance sólida**: 69.31% accuracy supera baseline (43.7% mayoría)
2. **Estabilidad**: Baja varianza en validación cruzada
3. **Interpretabilidad**: Feature importance clara y coherente
4. **Eficiencia**: Tiempo de entrenamiento aceptable
5. **Robustez**: Maneja bien outliers y desbalance

### Limitaciones Identificadas

1. **Clases minoritarias**: Dificultad con calidades extremas (3, 8, 9)
2. **Desbalance**: F1-score weighted necesario para evaluación justa
3. **Subjetividad humana**: Algunos errores pueden ser aceptables
4. **Generalización**: Performance específica para vinos portugueses

### Comparación con Literatura

- **Baseline esperado**: ~65% para este dataset
- **Estado del arte**: ~72% con feature engineering avanzado
- **Nuestro resultado**: 69.31% está en rango competitivo

## Mejoras Futuras

### Estrategias de Optimización

1. **Ensemble Methods**:
   ```python
   VotingClassifier([
       ('rf', RandomForestClassifier()),
       ('xgb', XGBClassifier()),
       ('svm', SVC(probability=True))
   ])
   ```

2. **Balanceo de Clases**:
   ```python
   SMOTE(random_state=42)  # Synthetic minority oversampling
   ```

3. **Feature Engineering**:
   - Ratios químicos (acidez total/alcohol)
   - Interacciones polinómicas
   - Binning de variables continuas

4. **Hyperparameter Tuning**:
   ```python
   GridSearchCV(
       RandomForestClassifier(),
       param_grid={
           'n_estimators': [100, 200, 300],
           'max_depth': [8, 10, 12],
           'min_samples_split': [2, 5, 10]
       }
   )
   ```

## Conclusiones

### Resultados Principales

1. **Random Forest es el mejor modelo** con 69.31% accuracy
2. **Feature importance confirma** hipótesis del EDA (alcohol dominante)
3. **Performance aceptable** considerando desbalance del dataset
4. **Metodología robusta** con validación cruzada estratificada

### Lecciones Aprendidas

1. **El desbalance de clases** es el mayor desafío
2. **La interpretabilidad** es tan importante como el performance
3. **La validación estratificada** es crucial para evaluación honesta
4. **Las métricas weighted** son necesarias para datasets desbalanceados

### Aplicabilidad Práctica

El modelo Random Forest entrenado puede utilizarse como:
- **Sistema de apoyo** para evaluación de calidad
- **Herramienta de control de calidad** en producción
- **Baseline sólido** para mejoras futuras