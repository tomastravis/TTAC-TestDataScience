# Dataset & EDA Analysis

## Dataset UCI Wine Quality

### Fuente y Características

El dataset Wine Quality es uno de los datasets más utilizados para problemas de clasificación en Machine Learning. Proviene del UCI ML Repository y contiene datos reales de evaluaciones de calidad de vinos portugueses.

**Información del Dataset:**
- **Fuente**: UCI Machine Learning Repository
- **Creador**: Paulo Cortez, University of Minho, Portugal
- **Año**: 2009
- **Licencia**: Creative Commons
- **URL**: https://archive.ics.uci.edu/ml/datasets/wine+quality

### Estructura de los Datos

#### Variables de Entrada (Features)
Todas las variables son numéricas y representan propiedades fisicoquímicas:

1. **fixed acidity**: Acidez fija (ácido tartárico en g/dm³)
2. **volatile acidity**: Acidez volátil (ácido acético en g/dm³)
3. **citric acid**: Ácido cítrico (g/dm³)
4. **residual sugar**: Azúcar residual (g/dm³)
5. **chlorides**: Cloruros (cloruro de sodio en g/dm³)
6. **free sulfur dioxide**: Dióxido de azufre libre (mg/dm³)
7. **total sulfur dioxide**: Dióxido de azufre total (mg/dm³)
8. **density**: Densidad (g/cm³)
9. **pH**: Potencial de hidrógeno
10. **sulphates**: Sulfatos (sulfato de potasio en g/dm³)
11. **alcohol**: Grado alcohólico (% por volumen)

#### Variable Objetivo (Target)
- **quality**: Calidad del vino (escala 0-10, valores observados 3-9)

### Estadísticas Descriptivas

#### Vinos Tintos (1,599 muestras)
```
Feature                  Mean    Std     Min     Max
fixed acidity           8.32    1.74    4.60   15.90
volatile acidity        0.53    0.18    0.12    1.58
citric acid             0.27    0.19    0.00    1.00
residual sugar          2.54    1.41    0.90   15.50
chlorides               0.09    0.05    0.01    0.61
free sulfur dioxide    15.87   10.46    1.00   72.00
total sulfur dioxide   46.47   32.89    6.00  289.00
density                 0.997   0.002   0.990   1.004
pH                      3.31    0.15    2.74    4.01
sulphates               0.66    0.17    0.33    2.00
alcohol                10.42    1.07    8.40   14.90
```

#### Vinos Blancos (4,898 muestras)
```
Feature                  Mean    Std     Min     Max
fixed acidity           6.85    0.84    3.80   14.20
volatile acidity        0.28    0.10    0.08    1.10
citric acid             0.33    0.12    0.00    1.66
residual sugar          6.39    5.07    0.60   65.80
chlorides               0.05    0.02    0.01    0.35
free sulfur dioxide    35.31   17.01    2.00  289.00
total sulfur dioxide  138.36   42.50    9.00  440.00
density                 0.994   0.003   0.987   1.039
pH                      3.19    0.15    2.72    3.82
sulphates               0.49    0.11    0.22    1.08
alcohol                10.51    1.23    8.00   14.20
```

## Análisis Exploratorio de Datos (EDA)

### 1. Distribución de la Variable Objetivo

La calidad del vino sigue una distribución aproximadamente normal centrada en valores medios:

- **Moda**: Calidad 6 (43.7% de las muestras)
- **Media**: 5.88
- **Mediana**: 6
- **Rango**: 3-9 (teóricamente 0-10)

### 2. Desbalance de Clases

El dataset presenta un **desbalance significativo** en las clases extremas:

```
Calidad  Muestras  Porcentaje
   3        30        0.5%      ← Clase minoritaria extrema
   4       216        3.3%
   5     2,138       32.9%
   6     2,836       43.7%      ← Clase mayoritaria
   7     1,079       16.6%
   8       193        3.0%
   9         5        0.1%      ← Clase minoritaria extrema
```

**Implicaciones**:
- Las clases 3 y 9 son extremadamente raras
- Modelos pueden tener sesgo hacia calidades medias (5-6)
- Se requieren métricas balanceadas (F1-score weighted)

### 3. Diferencias entre Vinos Tintos y Blancos

#### Principales Diferencias Químicas:
- **Acidez fija**: Tintos > Blancos (8.32 vs 6.85)
- **Acidez volátil**: Tintos > Blancos (0.53 vs 0.28)
- **Azúcar residual**: Blancos > Tintos (6.39 vs 2.54)
- **Sulfatos**: Tintos > Blancos (0.66 vs 0.49)
- **Densidad**: Tintos > Blancos (0.997 vs 0.994)

#### Distribución de Calidad por Tipo:
```
Tipo     Calidad Media  Std   Min  Max
Tinto         5.64      0.81   3    8
Blanco        5.88      0.89   3    9
```

### 4. Correlaciones Principales

#### Features más correlacionadas con calidad:
1. **alcohol**: +0.48 (correlación positiva fuerte)
2. **volatile acidity**: -0.39 (correlación negativa moderada)
3. **sulphates**: +0.25 (correlación positiva débil)
4. **citric acid**: +0.23 (correlación positiva débil)

#### Correlaciones entre features:
- **fixed acidity ↔ citric acid**: +0.67
- **fixed acidity ↔ density**: +0.67
- **total sulfur dioxide ↔ free sulfur dioxide**: +0.67
- **alcohol ↔ density**: -0.50

### 5. Detección de Outliers

#### Método IQR (Rango Intercuartílico):
- **residual sugar**: 547 outliers (8.4%)
- **total sulfur dioxide**: 340 outliers (5.2%)
- **free sulfur dioxide**: 298 outliers (4.6%)
- **chlorides**: 285 outliers (4.4%)

**Decisión**: Mantener outliers ya que pueden representar variaciones naturales en la producción vinícola.

### 6. Valores Faltantes

**Resultado**: El dataset está completo, sin valores faltantes en ninguna variable.

## Principales Insights del EDA

### Hallazgos Clave:
1. **El alcohol es el predictor más importante** de calidad
2. **Las calidades extremas son muy raras** (< 1% combinadas)
3. **Los vinos blancos dominan el dataset** (75.4%)
4. **Existe variabilidad química significativa** entre tipos de vino
5. **La acidez volátil afecta negativamente** la calidad percibida

### Implicaciones para el Modelado:
1. **Usar métricas balanceadas** debido al desbalance de clases
2. **Considerar estratificación** en train/test split
3. **El alcohol será probablemente** la feature más importante
4. **Evaluar modelos específicamente** en clases minoritarias
5. **Considerar ensemble methods** para mejorar predicción en clases raras

### Preparación de Datos:
1. **No se requiere tratamiento** de valores faltantes
2. **Escalado estándar recomendado** para SVM
3. **División estratificada 80/20** para mantener distribución
4. **Validación cruzada estratificada** para evaluación robusta