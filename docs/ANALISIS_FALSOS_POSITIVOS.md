# Análisis de falsos positivos del modelo XGBoost

## 1. Objetivo

Este análisis busca estudiar los falsos positivos generados por el modelo de clasificación binaria XGBoost entrenado sobre el dataset UNSW-NB15.

Un falso positivo ocurre cuando un flujo de tráfico normal es clasificado como un ataque. El objetivo es identificar patrones asociados a estos errores, evaluar el comportamiento de determinadas características y analizar las contribuciones del modelo mediante SHAP.

Los análisis se realizaron sobre datos de entrenamiento y validación cruzada. No se utilizaron los datos de test para ajustar el modelo ni seleccionar umbrales.

## 2. Experimentos de eliminación de características

Se realizaron experimentos de ablación eliminando individualmente y en conjunto las siguientes características:

- `sttl`
- `dttl`
- `ct_state_ttl`

Se evaluaron los modelos XGBoost y Regresión Logística, conservando el mismo esquema de validación cruzada para poder comparar los resultados.

Los experimentos no mostraron una mejora consistente que justificara eliminar estas variables. En XGBoost, quitar `dttl` produjo una reducción pequeña del FPR, mientras que eliminar las tres características simultáneamente no resolvió el problema. En Regresión Logística, las modificaciones tampoco mostraron una mejora general.

**Conclusión:** se conserva el modelo original con todas sus características. Los experimentos de ablación se consideran análisis exploratorios y no una mejora demostrada del modelo.

Los resultados individuales se encuentran en los archivos JSON de `reports/metrics/` y sus matrices de confusión y curvas ROC en `reports/figures/`.

## 3. Análisis de predicciones OOF

Se reconstruyeron las predicciones *out-of-fold* (OOF) de XGBoost para estudiar los errores sobre registros que no fueron utilizados para entrenar el modelo correspondiente a cada fold.

El umbral se seleccionó a partir de las predicciones OOF, buscando un recall mínimo de 0,90.

Resultados de la reconstrucción:

- Registros de entrenamiento: 175.341.
- Registros normales: 56.000.
- Falsos positivos: 4.489.
- Verdaderos negativos: 51.511.
- Umbral seleccionado: aproximadamente 0,3581.

El valor de umbral y las métricas pueden presentar pequeñas diferencias respecto del entrenamiento original debido a variaciones numéricas y no determinismo en la ejecución de XGBoost.

### 3.1. Relación con `sttl=254`

Se observó una concentración importante de falsos positivos en registros normales cuyo valor de `sttl` es 254:

- Aproximadamente el 91,53 % de los falsos positivos corresponde a registros con `sttl=254`.
- La tasa de falsos positivos dentro de ese grupo es aproximadamente 36,59 %.
- Fuera de ese grupo, la tasa de falsos positivos es aproximadamente 0,85 %.

Esto muestra una asociación descriptiva muy marcada entre `sttl=254` y los errores del modelo. No demuestra que esa característica sea la causa única de los falsos positivos.

## 4. Análisis de combinaciones de tráfico

Se profundizó el análisis sobre el tráfico normal con `sttl=254`, agrupándolo por combinaciones de protocolo, estado y servicio.

Se destacan los siguientes grupos:

| Combinación | Registros | Falsos positivos | Verdaderos negativos | FPR |
|---|---:|---:|---:|---:|
| TCP / FIN / sin servicio | 6.966 | 2.471 | 4.495 | 35,47 % |
| UDP / INT / sin servicio | 2.714 | 1.208 | 1.506 | 44,51 % |
| TCP / FIN / HTTP | 225 | 202 | 23 | 89,78 % |
| TCP / REQ | 856 | 0 | 856 | 0 % |

Las dos primeras combinaciones concentran una parte importante de los falsos positivos del grupo analizado. La combinación TCP / FIN / HTTP presenta una tasa elevada, pero debe interpretarse con cautela porque cuenta con solo 23 verdaderos negativos.

### 4.1. Diferencias descriptivas dentro de UDP / INT / sin servicio

Al comparar falsos positivos y verdaderos negativos se observaron diferencias en algunas variables:

- `ct_srv_dst`: mediana de 4 en falsos positivos y 20 en verdaderos negativos.
- `ct_srv_src`: mediana de 5 en falsos positivos y 20 en verdaderos negativos.
- `sbytes`: mediana de 104 en falsos positivos y 1.064 en verdaderos negativos.

Estas diferencias sugieren que el modelo encuentra patrones distintos dentro de este grupo, aunque por sí solas no permiten establecer causalidad.

## 5. Análisis de interpretabilidad con SHAP

Se utilizó SHAP para estudiar cómo contribuyen las características a las predicciones del modelo XGBoost.

El análisis se realizó sobre muestras de registros normales con `sttl=254`, separando falsos positivos y verdaderos negativos dentro de las combinaciones seleccionadas.

La muestra quedó conformada por:

| Grupo | Falsos positivos | Verdaderos negativos |
|---|---:|---:|
| TCP / FIN / sin servicio | 100 | 100 |
| UDP / INT / sin servicio | 100 | 100 |
| TCP / FIN / HTTP | 100 | 23 |
| **Total** | **300** | **223** |

La cantidad de registros por grupo está limitada por el muestreo realizado. En particular, el grupo TCP / FIN / HTTP tiene una cantidad reducida de verdaderos negativos.

### 5.1. Interpretación de los valores SHAP

Los valores SHAP obtenidos para XGBoost están expresados en la escala de margen o *log-odds*, no directamente en probabilidades.

- Un valor SHAP positivo empuja la predicción hacia la clase ataque.
- Un valor SHAP negativo empuja la predicción hacia la clase normal.
- `mean_shap` representa el promedio firmado de las contribuciones y permite observar su dirección.
- `mean_abs_shap` representa el promedio del valor absoluto y permite estimar la magnitud media de las contribuciones.

Por lo tanto, ambas métricas responden preguntas diferentes y no deben interpretarse como equivalentes.

### 5.2. Observaciones

En los tres grupos analizados, `sttl` presenta contribuciones positivas importantes tanto en falsos positivos como en verdaderos negativos. Esto es coherente con que el modelo asocia ese valor de la característica con la clase ataque, incluso en registros que en realidad son normales.

También aparecen diferencias en las contribuciones de variables como `sbytes`, `smean`, `ct_srv_dst` y `ct_srv_src`, dependiendo del grupo y del tipo de predicción.

En particular, dentro de UDP / INT / sin servicio, las contribuciones de variables relacionadas con el volumen de tráfico y la cantidad de conexiones ayudan a describir diferencias entre falsos positivos y verdaderos negativos.

Estas observaciones son descriptivas: SHAP explica cómo el modelo utiliza las características, pero no demuestra relaciones causales entre ellas y los errores.

## 6. Limitaciones

- La reconstrucción OOF puede no coincidir exactamente con métricas guardadas en ejecuciones anteriores debido a variaciones numéricas y no determinismo de XGBoost.
- El análisis SHAP utiliza una muestra limitada y estratificada, no la totalidad de los registros.
- El grupo TCP / FIN / HTTP tiene pocos verdaderos negativos, por lo que sus resultados son menos representativos.
- Las características pueden estar correlacionadas, lo que puede afectar cómo SHAP distribuye las contribuciones.
- Los análisis no demuestran causalidad ni garantizan que modificar una característica reduzca los falsos positivos.
- Los resultados corresponden a validación sobre datos de entrenamiento y no deben presentarse como una evaluación final sobre el conjunto de test.

## 7. Conclusiones

El análisis permitió identificar que los falsos positivos del modelo XGBoost no se distribuyen uniformemente: existe una concentración muy marcada en el tráfico normal con `sttl=254`.

Dentro de ese conjunto, las combinaciones TCP / FIN / sin servicio y UDP / INT / sin servicio reúnen una cantidad importante de errores. También se encontraron diferencias descriptivas en variables relacionadas con el volumen y la frecuencia de conexiones.

El análisis SHAP mostró que `sttl` tiene una contribución positiva relevante hacia la predicción de ataque, incluso en registros normales que terminan clasificados incorrectamente. Otras variables también presentan contribuciones diferentes entre falsos positivos y verdaderos negativos.

Por otra parte, los experimentos de eliminación de características no demostraron una mejora consistente, por lo que se decidió conservar el modelo original.

En conjunto, estos análisis permiten comprender mejor el comportamiento del clasificador y caracterizar algunos de sus errores, sin afirmar que se haya encontrado una solución definitiva al problema de falsos positivos.
