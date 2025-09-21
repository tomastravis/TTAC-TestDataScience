# Tests AI & Data Science
## For Hybrid Intelligence Spain
**Version 0.1 - 2024**

---

## INTRODUCCIÓN

El objetivo de estas pruebas es el de evaluar las capacidades, conocimientos y habilidades del candidato/a en materia de inteligencia artificial y ciencia de datos.

### Requisitos Técnicos

- **Lenguaje de programación**: Todas las pruebas deberán ser resueltas utilizando **Python** como lenguaje de programación, en particular con una versión estable. 

--> Se ha elegido 3.12 al ser la mas reciente y al ser version estable.

- **Reproducibilidad**: El candidato/a deberá poder garantizar la reproducibilidad del análisis realizado en cada prueba mediante la definición de dependencias a través de fichero `requirements` / `poetry`.

--> Usar Poetry con pyproject.toml para gestión de dependencias y versiones exactas.  
--> Incluir deployment/requirements.txt para compatibilidad con entornos sin Poetry.

- **Calidad de código**: El candidato/a deberá seguir las reglas de estilo Python PEP8 así como utilizar analizadores de código para asegurar la calidad de este:
  - Código muerto
  - Tipado de funciones
  - Optimización de imports
  - Complejidad ciclomática

--> Estrategia: Configurar black, flake8, mypy, isort en pyproject.toml con hooks pre-commit.  
--> Usar type hints obligatorios y límites estrictos de complejidad ciclomática (max 10).  
--> Justificación vs alternativas:
   • black vs autopep8: black es más opinionado, elimina debates de formato
   • flake8 vs pylint: flake8 más rápido, menos falsos positivos en ML
   • mypy vs pyre: mypy mejor integrado en ecosistema Python científico
   • pre-commit vs IDE hooks: garantiza consistencia independiente del editor

### Jupyter Notebooks

- Los jupyter notebooks deberán estar **ordenados y documentados** indicando claramente el orden de ejecución y la estructura de estos mediante celdas markdown con títulos y subtítulos.

--> Estrategia: Crear notebooks con numeración clara (01_, 02_, etc.) y headers markdown H1-H3.  
--> Incluir tabla de contenidos y celdas explicativas entre cada sección de código.  
--> Justificación vs alternativas:
   • Numeración vs nombres descriptivos: números garantizan orden de ejecución
   • H1-H3 vs formato libre: estructura jerárquica facilita navegación
   • Celdas explicativas vs comentarios código: mejor legibilidad y documentación

- Se recomienda **exportar el jupyter notebook** con el resultado embebido a formato HTML.

--> Estrategia: Configurar exportación automática a HTML en run_analysis.py tras completar análisis.  
--> Incluir outputs ejecutados y gráficos embebidos para evaluación offline.  
--> Justificación vs alternativas:
   • HTML vs PDF: HTML preserva interactividad y es más ligero
   • Automático vs manual: reduce errores humanos y garantiza consistency
   • Embebido vs separado: facilita evaluación sin dependencias externas

### Estructura del Proyecto

- La estructura de los entregables deberá ser la de un **repositorio de código tipo cookie cutter** con un fichero `README.md` indicando la función de cada módulo/script/fichero del entregable.

--> Estrategia: Usar estructura src/ modular con ttac_test_ds1/ como package principal.  
--> README.md detallado con descripción de cada directorio y archivo clave.  
--> Justificación vs alternativas:
   • src/ vs root package: separa código fuente de configuración y docs
   • Modular vs monolítico: facilita testing, reutilización y mantenimiento
   • Package vs scripts sueltos: permite imports limpos y distribución

### Valoraciones Positivas

- **Presentación**: Se valorará de forma positiva la entrega de una presentación con el planteamiento del caso, resultados y conclusiones obtenidos o un informe en LaTeX/beamer (en ambos casos solo entregar el PDF generado).

--> Estrategia: Crear presentación PDF final resumiendo metodología, hallazgos y conclusiones.  
--> Usar formato profesional con gráficos claros y métricas clave destacadas.  
--> Justificación vs alternativas:
   • PDF vs PowerPoint: universalmente accesible, mantiene formato
   • Resumen vs detalle completo: facilita evaluación rápida ejecutiva
   • Gráficos propios vs screenshots: mejor calidad y profesionalidad

- **Documentación**: Se valorará de forma positiva la creación de documentación con **MkDocs**.

--> Estrategia: Implementar MkDocs básico si tiempo permite, priorizando código funcional.  
--> Documentar funciones principales con docstrings detallados como alternativa robusta.  
--> Justificación vs alternativas:
   • MkDocs vs Sphinx: más simple y enfocado en markdown
   • Básico vs completo: time-boxing para no sacrificar funcionalidad core
   • Docstrings vs wiki separado: documentación vive con el código

### Entrega

#### Opción 1: Email
Las entregas se realizarán vía email mediante un archivo comprimido (uno por cada prueba) cuyo nombre siga la nomenclatura:
```
XYZ-TestDataScience-W.zip
```
Donde:
- `XYZ` = iniciales del candidato/a
- `W` = número de prueba

#### Opción 2: GitHub
Alternativamente se podrá realizar la entrega mediante un **repositorio público de GitHub** aplicando la misma nomenclatura.

---

## TEST 1 – DATA SCIENCE - CLASIFICACIÓN

### Objetivos

1. **Dataset**: Escoger un dataset abierto de clasificación que **no sea el de Iris** (referenciar fuente)

--> Estrategia: Usar Wine Quality Dataset de UCI ML Repository con documentación completa de fuente.  
--> Dataset balanceado con features numéricos y target de calidad para clasificación multiclase.  
--> Justificación vs alternativas:
   • Wine vs Titanic: evita dataset sobreusado, mantiene complejidad adecuada
   • UCI vs Kaggle: fuente académica más seria, datos verificados
   • Multiclase vs binario: demuestra manejo de complejidad superior
   • Numérico vs mixto: evita complejidad de encoding categórico innecesaria

2. **Análisis Exploratorio (EDA)**: En un Jupyter notebook realizar análisis exploratorio, incluyendo:
   - Limpieza de datos
   - Transformaciones
   - Agregaciones
   - Visualizaciones que se consideren oportunas

--> Estrategia: Notebook estructurado 01_eda.ipynb con análisis completo de distribuciones, correlaciones, outliers.  
--> Incluir análisis estadístico descriptivo y visualizaciones con matplotlib/seaborn profesionales.  
--> Justificación vs alternativas:
   • Notebook vs script: mejor para EDA interactivo y visualización
   • matplotlib/seaborn vs plotly: más estable, mejor integración con exportación
   • Estadístico formal vs exploratorio básico: demuestra rigor científico
   • Outliers analysis vs ignorar: crítico para calidad del modelo

3. **Modelado**: 
   - Seleccionar, entrenar y testear el/los modelo/s que se consideren apropiados
   - Justificar el modelo elegido en función de métricas de rendimiento

--> Estrategia: Comparar múltiples algoritmos (Random Forest, SVM, XGBoost) con validación cruzada.  
--> Usar métricas apropiadas (accuracy, precision, recall, F1) con justificación estadística.  
--> Justificación vs alternativas:
   • Multiple vs single model: demuestra conocimiento amplio y evita overfitting
   • RF/SVM/XGB vs DL: apropiado para dataset tabular de tamaño medio
   • Cross-validation vs train/test: más robusto estadísticamente
   • Múltiples métricas vs accuracy: esencial para clasificación desbalanceada

4. **Conclusiones**: Elaborar conclusiones del ejercicio llevado a cabo

--> Estrategia: Sección dedicated en notebook final con insights de negocio y limitaciones del modelo.  
--> Incluir recomendaciones para mejoras futuras y aplicabilidad práctica.  
--> Justificación vs alternativas:
   • Business insights vs solo técnico: demuestra visión comercial
   • Limitaciones explícitas vs solo resultados: honestidad profesional
   • Recomendaciones vs conclusiones finales: valor añadido consultivo
   • Aplicabilidad vs teoría: enfoque práctico orientado a producción

5. **Entregable**: Preparar entregable con todos los ficheros necesarios para:
   - Reproducir el análisis
   - Poner en producción el modelo entrenado (integración DevOps)

--> Estrategia: Scripts de deployment básicos para training/inference y serialización de modelos.  
--> Incluir script de entrenamiento, script de predicción y modelo serializado con joblib.  
--> Justificación vs alternativas:
   • Scripts vs manual: automatización reduce errores y facilita reproducibilidad
   • joblib vs pickle: mejor para objetos numpy/sklearn, más eficiente
   • Modular vs monolítico: separación clara entre training e inference
   • Batch inference vs interactivo: cumple requisito sin complejidad innecesaria

---

## TEST 2 – TIME SERIES - REGRESSION

### Objetivos

1. **Dataset**: Escoger un dataset abierto de series temporales multivariante que:
   - **No sea financiero**
   - La variable a predecir **no tenga estacionalidad** (referenciar fuente)

--> Estrategia: Usar dataset de calidad del aire o temperatura industrial con múltiples sensores.  
--> Verificar no-estacionalidad con tests ADF (Augmented Dickey-Fuller) y KPSS.  
--> Justificación vs alternativas:
   • Calidad aire vs financiero: cumple requisito y tiene relevancia social
   • Industrial vs meteorológico: más estable, menos patrones estacionales
   • Tests estadísticos vs inspección visual: rigor científico verificable

2. **Verificación**: Realizar comprobación de que la serie temporal es **no-estacional**

--> Estrategia: Aplicar batería de tests estadísticos y análisis de componentes.  
--> Documentar todos los p-values y estadísticos para transparencia completa.  
--> Justificación vs alternativas:
   • Múltiples tests vs single test: mayor confianza estadística
   • Documenting p-values vs conclusiones: permite verificación independiente
   • Decomposición vs tests solo: visualización complementa análisis

3. **Modelado**: Seleccionar, entrenar y testear el/los modelo/s de machine learning/series temporales que permitan:
   - **Predecir 100 períodos de tiempo en el futuro** con la particularidad que a partir del instante donde se inicia la predicción no se dispone de los valores de las variables regresoras

--> Estrategia: Implementar LSTM/GRU para capturar dependencias temporales largas.  
--> Usar técnicas de windowing y lag features para compensar falta de regresores.  
--> Justificación vs alternativas:
   • LSTM vs ARIMA: mejor para relaciones no-lineales multivariantes
   • Deep Learning vs Prophet: más flexible sin asumir componentes estacionales
   • Windowing vs autoregresivo simple: captura más contexto temporal
   • 100 períodos: requiere modelos robustos a drift temporal

4. **Evaluación**:
   - Representar gráficamente los resultados
   - Evaluar la bondad del modelo utilizando los KPIs adecuados

--> Estrategia: Usar RMSE, MAE, MAPE y análisis de residuos con gráficos tiempo-serie.  
--> Incluir intervalos de confianza y análisis de uncertainty.  
--> Justificación vs alternativas:
   • Múltiples métricas vs single KPI: perspectiva completa del rendimiento
   • Time-series plots vs scatter: preserva información temporal crítica
   • Confidence intervals vs point estimates: esencial para predicciones largas
   • Residual analysis vs métricas solo: detecta patrones no capturados

5. **Conclusiones**: Elaborar conclusiones del ejercicio llevado a cabo

--> Estrategia: Analizar performance vs benchmark y discutir aplicabilidad práctica.  
--> Incluir análisis de limitaciones y escenarios donde el modelo falla.  
--> Justificación vs alternativas:
   • Benchmark vs absolute metrics: contextualiza rendimiento real
   • Failure analysis vs solo successes: demuestra comprensión profunda
   • Practical applicability vs academic: enfoque empresarial

6. **Entregable**: Preparar entregable con todos los ficheros necesarios para:
   - Reproducir el análisis
   - Poner en producción el/los modelo/s entrenado/s (integración DevOps)

--> Estrategia: Scripts de deployment básicos para training/inference y serialización de modelos.  
--> Incluir script de entrenamiento, script de predicción y modelo serializado con joblib.  
--> Justificación vs alternativas:
   • Scripts vs manual: automatización reduce errores y facilita reproducibilidad
   • joblib vs pickle: mejor para objetos numpy/sklearn, más eficiente
   • Modular vs monolítico: separación clara entre training e inference
   • Batch inference vs interactivo: cumple requisito sin complejidad innecesaria

---

## Notas Adicionales

> **Accelerator for prototyping decision-making models in data-driven business**  
> Hybrid Intelligence Spain – J-A. Velasco, F. Bermejo

---

**Company Confidential © Capgemini 2024. All rights reserved**
