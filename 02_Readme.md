Pipeline de Regresión Lineal con PCA y Optimización de Residuales
1. Descripción General
Este script implementa un pipeline completo de análisis de datos y modelado predictivo en Python. El objetivo principal es predecir una variable Objetivo utilizando un conjunto de características complejas.

El proceso se desarrolla en tres etapas fundamentales:

Ingeniería de Características y Reducción de Dimensionalidad: El script realiza un preprocesamiento avanzado que incluye el escalado y la combinación de variables constantes. Aplica de forma separada el Análisis de Componentes Principales (PCA) a dos subconjuntos de características (temporales y otras) para reducir la dimensionalidad y capturar la varianza principal.

Modelado Predictivo Base: Se entrena un modelo de Regresión Lineal utilizando un sklearn.pipeline.Pipeline que incluye escalado (StandardScaler). El modelo se evalúa rigurosamente mediante validación cruzada (K-Fold de 10 divisiones) para estimar su rendimiento (RMSE y R²) y, posteriormente, se prueba en un conjunto de test separado.

Optimización y Ajuste de Residuales: Los errores (residuales) generados por el modelo de regresión lineal base se someten a un proceso de optimización. Utilizando scipy.optimize.minimize con el método L-BFGS-B, el script calcula una matriz de ajuste H que minimiza una función de coste regularizada (norma de Frobenius). Este ajuste corrige patrones sistemáticos en los errores, generando una predicción final ajustada (Adjusted_Pred).

El script genera visualizaciones de diagnóstico (análisis de residuales, comparativas de predicción) tanto para el modelo base como para el modelo ajustado, y guarda los resultados completos en un archivo CSV.

2. Metodología Detallada
2.1. Preprocesamiento y PCA
El feature engineering es un paso crucial:

Variables Constantes: Las variables definidas en constant_vars se agregan multiplicativamente (prod). El resultado (Product_Constantes) se normaliza usando StandardScaler.

PCA Temporal: Las variables en temporal_vars se escalan y se reducen a n_PCA_temporal componentes principales.

PCA Otras Variables: El resto de las características (excluyendo la variable Objetivo y las ya procesadas) se escalan y reducen a n_PCA_others componentes.

Conjunto Final de Predictores (X): El conjunto de datos para el modelo se construye concatenando:

Componentes PCA temporales (PCA_Temporal_*).

Componentes PCA de otras variables (PCA_Others_*).

Componentes PCA originales (si existen, p.ej., PC*).

El producto escalado de variables constantes (Scaled_Product_Constantes).

2.2. Modelo de Regresión Lineal Base
Se utiliza un Pipeline de Scikit-learn que garantiza que el escalado se aplique correctamente durante la validación cruzada y el entrenamiento final.

Métricas de Evaluación:

RMSE (Root Mean Squared Error): rmse(y_true, y_pred)

R² (Coeficiente de Determinación): r2_score

2.3. Optimización de Residuales
Esta etapa refina las predicciones del modelo lineal.

Sea E el vector de residuales del modelo base, E=Y 
real
​
 −Y 
pred_lineal
​
 .

Buscamos una matriz de ajuste H (de la misma dimensión que E) que minimice la siguiente función de coste L(H):

L(H)=∑(E+H) 
2
 +λ∣∣H∣∣ 
F
2
​
 
Donde:

(E+H) es el residual ajustado.

∑(E+H) 
2
  es la suma de los cuadrados de los residuales ajustados (término de error).

∣∣H∣∣ 
F
2
​
  es el cuadrado de la norma de Frobenius de H. Actúa como un término de regularización L2 que penaliza ajustes H de gran magnitud, previniendo el sobreajuste.

λ es el hiperparámetro de regularización (lambda_reg).

La función optimize_H utiliza scipy.optimize.minimize (con el método 'L-BFGS-B') para encontrar la H 
optimal
​
  que minimiza L(H).

Las predicciones y errores finales se calculan como:

Adjusted_Error = E+H 
optimal
​
 

Adjusted_Pred = Y 
real
​
 −Adjusted_Error

3. Requisitos e Instalación
3.1. Dependencias
El script requiere las siguientes bibliotecas de Python. Se recomienda ejecutarlo en un entorno virtual.

pandas

numpy

scikit-learn

matplotlib

scipy

3.2. Instalación
Clone o descargue el repositorio.

Cree y active un entorno virtual (opcional pero recomendado):

Bash

python -m venv venv
# En Windows
.\venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
Instale las bibliotecas necesarias:

Bash

pip install pandas numpy scikit-learn matplotlib scipy
4. Instrucciones de Uso
Para utilizar el script, es fundamental configurar las rutas de los archivos y verificar que las listas de variables coincidan con su conjunto de datos.

Paso 1: Configurar el Script
Abra el archivo .py y modifique las siguientes variables dentro de la función lineal_regresion_PCA():

Ruta del Archivo de Entrada (Línea 18):

Modifique file_path para que apunte a su archivo CSV de entrada.

Python

file_path = '/content/drive/MyDrive/Collab/_TERMINAL - Proyecto Cono/02. Dades/BTC_KH22_TSNE1h_cortado.csv'
Ruta del Archivo de Salida (Línea 219):

Modifique output_file_path para definir dónde se guardarán los resultados.

Python

output_file_path = '/content/drive/ubicacion_datos.csv'
Listas de Variables (Crítico):

Asegúrese de que las listas constant_vars (línea 25) y temporal_vars (línea 33) coincidan exactamente con los nombres de las columnas en su archivo CSV. El script deduce automáticamente las other_vars excluyendo las de estas listas y la variable Objetivo.

Parámetros del Modelo (Opcional):

Puede ajustar el número de componentes PCA a retener (líneas 21-22):

Python

n_PCA_temporal = 12
n_PCA_others = 12
Puede ajustar el parámetro de regularización lambda_reg en la llamada a optimize_H (línea 160).

Paso 2: Ejecutar el Script
Una vez guardados los cambios de configuración, ejecute el script desde su terminal:

Bash

python nombre_del_script.py
(Reemplace nombre_del_script.py con el nombre de su archivo).

5. Ejemplo de Salida (Uso Final)
Al ejecutar el script, obtendrá los siguientes resultados:

5.1. Salida en Consola
El script imprimirá un informe detallado del rendimiento del modelo, incluyendo:

Las puntuaciones RMSE y R² para cada fold de la validación cruzada.

El promedio de RMSE y R² de la validación cruzada.

El RMSE y R² finales obtenidos en el conjunto de prueba.

Las métricas RMSE y R² del modelo ajustado (calculadas sobre el total de los datos).

RMSE de validación cruzada: [0.0125 0.0130 0.0122 ...]
RMSE promedio de validación cruzada: 0.0126
R² de validación cruzada:   [99.78 99.76 99.79 ... ]%
R² promedio de validación cruzada: 99.78%
RMSE en el conjunto de prueba: 0.0124
R² en el conjunto de prueba: 99.79%

... (Resultados de la optimización) ...

RMSE Ajustado: 0.0122
R² Ajustado: 99.80%
Datos guardados en: /content/drive/ubicacion_datos.csv
5.2. Visualizaciones
Aparecerán secuencialmente cinco ventanas de matplotlib mostrando los siguientes gráficos de diagnóstico:

Análisis de Residuales (Modelo Base): Muestra los residuales del modelo lineal frente a los valores reales.

Distribución del Error de Predicción (Modelo Base): Histograma de los residuales del modelo lineal.

Valores Reales vs Predicciones (Modelo Base): Compara las predicciones del modelo lineal con los valores reales.

Análisis de Residuales (Ajustados): Muestra los residuales ajustados (post-optimización) frente a los valores reales.

Valores Reales vs Predicciones (Ajustados): Compara las predicciones ajustadas con los valores reales.

5.3. Archivo CSV de Salida
Se generará un archivo CSV en la ruta output_file_path. Este archivo contendrá:

Date (como índice)

Unix

Objetivo (el valor real)

Todas las componentes PCA generadas (PCA_Temporal_*, PCA_Others_*).

Cualquier componente PC original que existiera.

Scaled_Product_Constantes

Lineal_Pred (Predicción del modelo base)

Error (Error del modelo base: Objetivo - Lineal_Pred)

Adjusted_Pred (Predicción ajustada tras la optimización)

Adjusted_Error (Error ajustado: Objetivo - Adjusted_Pred)

Este archivo de salida está listo para un análisis más profundo o para ser utilizado en etapas posteriores del proyecto.
