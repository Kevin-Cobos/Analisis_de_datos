# `README.md`: Script 1 (v4.0) - Pre-procesador PCA y Analizador de Error

## 1\. Descripción General

Este script implementa la **primera etapa** de un pipeline de modelado en dos partes. Su propósito **no** es generar el modelo predictivo final, sino tomar datos crudos de alta dimensionalidad y realizar un pre-procesamiento avanzado y una reducción de dimensionalidad optimizada.

La salida principal de este script (`BTC_KH22_TSNE1h_cortado.csv`) está diseñada para ser el **archivo de entrada directo** para el "Script 2" (el modelo final).

### Flujo de Trabajo

1.  **Carga:** Lee un archivo de datos crudos (`raw_data_input.csv`) que contiene todas las características (temporales, constantes, objetivo y características crudas de alta dimensionalidad).
2.  **Identificación y Separación:** Divide inteligentemente las columnas en dos grupos:
      * **Características Crudas (para S1):** Columnas identificadas por un prefijo (p.ej., `Raw_Feature_...`).
      * **Características "Passthrough" (para S2):** Todas las demás columnas (`Objetivo`, `Unix`, `temporal_vars`, `constant_vars`, `other_vars`) que el Script 2 necesita para *su propio* análisis.
3.  **PCA (Nivel 1):** Aplica un Análisis de Componentes Principales (PCA) optimizado *exclusivamente* a las "Características Crudas". Esto genera las columnas `PC1`, `PC2`, ..., `PCk`.
4.  **Análisis de Error (Nivel 1):** Entrena un modelo de regresión lineal simple (`Objetivo ~ PC*`) para generar las columnas `Error_Lineal` y `Lineal_Pred`. Estas columnas se convierten en *nuevas características* para el Script 2.
5.  **Análisis Avanzado:** Realiza un análisis de correlación estático y dinámico (vectorizado) para diagnosticar la relación entre las nuevas `PC*` y el `Error_Lineal`.
6.  **Salida:** Combina las características "Passthrough" con las nuevas `PC*` y `Error_Lineal` en un único archivo (`BTC_KH22_TSNE1h_cortado.csv`).

## 2\. Metodología y Características Clave

Este script implementa optimizaciones de alto rendimiento y análisis rigurosos:

  * **PCA Optimizado (SVD Aleatorizado):** Utiliza la estrategia `randomized` ($O(N \cdot k^2)$) por defecto, en lugar del SVD completo ($O(N \cdot P^2)$), para manejar eficientemente conjuntos de datos con un gran número de características crudas ($P \gg k$).
  * **Determinación de $k$ (Método del Codo):** Utiliza la biblioteca `kneed` para encontrar el número óptimo de componentes ($k$) basándose en el punto de máxima inflexión de los *eigenvalues*.
  * **Vectorización Robusta ($O(N \cdot k)$):** El análisis de correlación dinámica (ventana móvil) se implementa con `numpy` puro, utilizando `cumsum` para evitar bucles (`pandas.rolling.corr()`), logrando un *speedup* teórico de factor $W$ (tamaño de la ventana).
  * **Separación de Responsabilidades:** El script nunca modifica las `temporal_vars` o `constant_vars`; simplemente las "pasa a través", respetando el dominio de procesamiento del Script 2.

## 3\. Requisitos e Instalación

El script requiere las siguientes bibliotecas de Python:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib`
  * `kneed` (para el método del codo)
  * `pyarrow` o `fastparquet` (para guardar los artefactos de análisis)

**Instalación:**

```bash
pip install pandas numpy scikit-learn matplotlib kneed pyarrow
```

## 4\. Arquitectura de Archivos (Entrada y Salida)

Esta es la sección más importante para el uso correcto del pipeline.

### 4.1. Archivo de Entrada (Input)

El script espera **un** archivo CSV definido en la variable `INPUT_DATA_FILE`.

  * **Nombre Esperado:** `raw_data_input.csv`
  * **Formato Requerido:**
      * Debe ser un archivo `.csv`.
      * Debe tener un **índice de tipo Datetime** llamado `Date`.
      * Debe contener **todas** las columnas que Script 2 necesita (p.ej., `Objetivo`, `Unix`, 'Hour', 'Day', 'G\_constant', etc.).
      * Debe contener las características crudas de alta dimensionalidad (p.ej., `Raw_Feature_1`, `Raw_Feature_2`, ...).

### 4.2. Archivo de Salida Principal (Output)

El script genera **un** archivo CSV principal, que es la entrada para Script 2.

  * **Nombre Generado:** `BTC_KH22_TSNE1h_cortado.csv`
  * **Contenido:**
      * Todas las columnas de `raw_data_input.csv` **EXCEPTO** las `Raw_Feature_*`.
      * **Nuevas Columnas (de PCA):** `PC1`, `PC2`, ..., `PCk`
      * **Nuevas Columnas (de PCR):** `Error_Lineal`, `Lineal_Pred`

### 4.3. Artefactos de Análisis (Salidas Secundarias)

El script también genera los siguientes archivos para diagnóstico (guardados en formato `.parquet` por eficiencia):

  * `script_1_preprocessor.log`: Log detallado de cada paso de ejecución.
  * `pca_diagnostics_plot.png`: Gráfico del Método del Codo y Varianza Explicada.
  * `script_1_static_correlation.parquet`: Matriz de correlación (`PC*` vs `Error_Lineal`).
  * `script_1_dynamic_corr_target.parquet`: Correlación dinámica (`PC*` vs `Objetivo`).
  * `script_1_dynamic_corr_error.parquet`: Correlación dinámica (`PC*` vs `Error_Lineal`).

## 5\. Instrucciones de Uso

### Paso 1: Preparar el Entorno

Asegúrese de tener todas las bibliotecas listadas en la Sección 3 instaladas.

### Paso 2: Preparar los Datos de Entrada

1.  Cree su archivo de datos crudos.

2.  Asegúrese de que el archivo:

      * Se llame `raw_data_input.csv` y esté en el mismo directorio que el script.
      * Tenga un índice de fecha llamado `Date`.
      * Contenga las columnas `Objetivo`, `Unix`, y todas las `temporal_vars` y `constant_vars`.
      * Contenga sus características crudas con el prefijo `Raw_Feature_`.

    *Nota: Si el archivo `raw_data_input.csv` no existe, el script (al ejecutarse) generará automáticamente un archivo de demostración con datos simulados.*

### Paso 3: Configurar el Script (Opcional)

Si sus datos crudos tienen un prefijo diferente a `Raw_Feature_`, modifique la variable global:

```python
# SECCIÓN 3: DEFINIR VARIABLES GLOBALES Y CONFIGURACIÓN
...
RAW_FEATURE_PREFIX = 'Mi_Prefijo_Crudo_'
...
```

### Paso 4: Ejecutar el Script

Abra una terminal en el directorio del script y ejecute:

```bash
python script_1_v4.py
```

*(Asumiendo que ha guardado el script como `script_1_v4.py`)*

El script se ejecutará, generará los logs, los gráficos y, lo más importante, el archivo `BTC_KH22_TSNE1h_cortado.csv`.

## 6\. Flujo del Pipeline (Resumen)

1.  Ejecute este script (`script_1_v4.py`) para procesar `raw_data_input.csv`.
2.  Verifique que `BTC_KH22_TSNE1h_cortado.csv` se haya creado.
3.  **Inmediatamente después**, ejecute el "Script 2 (v4.0)", que cargará automáticamente este archivo CSV para construir el modelo final.
