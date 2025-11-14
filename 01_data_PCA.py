"""
================================================================================
SCRIPT 1 (v4.0 - Versión Final Compatible): PREPROCESADOR PCA Y ANALIZADOR DE ERROR
================================================================================

(Script Python Modular - Buenas prácticas y notación Big-O optimizada)

Descripción:
Versión final y compatible del Script 1, diseñada para actuar como el
pre-procesador principal para el "Script 2" (lineal_regresion_PCA).

Flujo de Trabajo:
1.  Carga datos crudos (se espera 'raw_data_input.csv').
2.  Identifica y separa las columnas:
    a. (Passthrough): Columnas que 'Script 2' espera (Temporales, Constantes,
       Otras vars, Objetivo, Unix).
    b. (PCA Input): Columnas de características crudas (prefijo 'Raw_Feature_').
    c. (Target/Time): Columnas de Objetivo y Tiempo.
3.  Aplica PCA optimizado (SVD Aleatorizado O(N*k^2)) *solo* al grupo (b)
    para generar las columnas 'PC*' (las 'pca_original_columns' de Script 2).
4.  Realiza la regresión (PCR) 'Objetivo ~ PC*' para generar 'Error_Lineal'.
5.  Realiza el análisis de correlación estático y dinámico (vectorizado O(N*k)).
6.  Guarda el archivo de salida principal como 'BTC_KH22_TSNE1h_cortado.csv',
    conteniendo (a), (c), las nuevas 'PC*', y 'Error_Lineal',
    listo para ser consumido por "Script 2".
"""

# SECCIÓN 1: IMPORTAR BIBLIOTECAS
import sys
import os
import logging
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from kneed import KneeLocator
import warnings
import gc

# --- Configuración de Advertencias ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# SECCIÓN 2: DEFINIR DIRECTORIOS Y UBICACIONES
BASE_DIR = os.getcwd()

# --- Entrada (Datos Crudos) ---
# Se espera un archivo CSV con un índice 'Date'
INPUT_DATA_FILE = os.path.join(BASE_DIR, 'raw_data_input.csv')

# --- Salida Principal (Input para Script 2) ---
# ¡¡CRÍTICO!!: Esta ruta y formato (.csv) deben coincidir EXACTAMENTE
# con el 'file_path' esperado por el Script 2.
OUTPUT_DATA_FILE = os.path.join(
    BASE_DIR, 'csv_output.csv'
)

# --- Salidas de Análisis (Artefactos de este script, en Parquet) ---
OUTPUT_LOG_FILE = os.path.join(BASE_DIR, 'script_1_preprocessor.log')
OUTPUT_PCA_DIAGNOSTICS_PLOT = os.path.join(BASE_DIR, 'pca_diagnostics_plot.png')
OUTPUT_STATIC_CORR_FILE = os.path.join(
    BASE_DIR, 'script_1_static_correlation.parquet'
)
OUTPUT_DYNAMIC_CORR_TARGET_FILE = os.path.join(
    BASE_DIR, 'script_1_dynamic_corr_target.parquet'
)
OUTPUT_DYNAMIC_CORR_ERROR_FILE = os.path.join(
    BASE_DIR, 'script_1_dynamic_corr_error.parquet'
)


# SECCIÓN 3: DEFINIR VARIABLES GLOBALES Y CONFIGURACIÓN

# --- Variables Clave ---
TARGET_COLUMN = 'Objetivo'
TIME_COLUMN = 'Unix'

# --- Columnas 'Passthrough' (Requeridas por Script 2) ---
# Estas columnas NO serán usadas por el PCA de este script.
# Se copian exactamente desde el Script 2 para asegurar la compatibilidad.
PASSTHROUGH_COLS_CONSTANT = [
    'Frecuencia Grupal', 'C_Rel_Luz', 'Frecuencia Fundamental General',
    'G_constant', 'Fundamental_Frequency', 'inc_t'
]
PASSTHROUGH_COLS_TEMPORAL = [
    'Minutes', 'Hour', 'Day', 'DayOfWeek', 'DayOfYear', 'Week',
    'WeekOfYear', 'Month', 'Year', 'sin_t', 'Model_dynamic_temporal',
    'Model_static_temporal', 'Model_large_temporal',
    'Model_total_temporal', 'Model_combined', 'Model_combined_normalized',
    'Seconds'
]

# --- Configuración de PCA (Optimizado) ---
RAW_FEATURE_PREFIX = 'Raw_Feature_' # Prefijo de las columnas a comprimir
PCA_STRATEGY = 'randomized' # O(N*k^2)
PCA_N_COMPONENTS_ESTIMATE = 100 # Componentes a probar en modo 'randomized'

# --- Configuración de Análisis ---
DYNAMIC_WINDOW_SIZE = 100

# Configurar el logger global
logging.basicConfig(level=logging.DEBUG)

# SECCIÓN 4: DEFINIR FUNCIONES MODULARES POR SUB-PROCESOS

def setup_logging(log_file: str) -> None:
    """Configura el sistema de logging (archivo y consola)."""
    # O(1)
    logger = logging.getLogger()
    logger.handlers = []
    log_format = logging.Formatter(
        '%(asctime)s - [%(levelname)-8s] - (%(funcName)s) - %(message)s'
    )
    # Handler Archivo (DEBUG)
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except IOError as e:
        print(f"Error crítico: No se puede escribir en log {log_file}. {e}")
    # Handler Consola (INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logging.info("Sistema de logging configurado.")

class DataValidationError(Exception):
    """Excepción personalizada para errores de validación de datos."""
    pass

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Carga el archivo CSV de entrada (datos crudos).
    Espera un índice 'Date'.
    """
    # O(N*M) - Limitado por I/O
    logging.info(f"Cargando datos crudos desde: {filepath}")
    
    if not os.path.exists(filepath):
        logging.error(f"Archivo no encontrado: {filepath}")
        raise DataValidationError(f"Archivo no encontrado: {filepath}")

    try:
        # Script 2 usa 'index_col='Date'', así que lo leemos igual.
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        
    except pd.errors.ParserError as e:
        logging.error(f"Error de parseo en el CSV: {e}")
        raise DataValidationError(f"Error de parseo en el CSV: {e}")
    except KeyError:
        logging.error("Error: El archivo CSV de entrada no tiene "
                      "una columna/índice 'Date'.")
        raise DataValidationError("Índice 'Date' no encontrado en el CSV crudo.")
    except Exception as e:
        logging.error(f"Error inesperado al cargar los datos: {e}")
        raise

    if df.empty:
        logging.error("El archivo CSV está vacío.")
        raise DataValidationError("El archivo CSV cargado está vacío.")
        
    logging.info(f"Datos cargados exitosamente. Forma: {df.shape}")
    return df

def identify_columns_for_pipeline(
    df: pd.DataFrame, 
    target_col: str, 
    time_col: str, 
    passthrough_lists: List[List[str]],
    raw_feature_prefix: str
) -> Tuple[pd.Series, Optional[pd.Series], pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame en los 4 componentes requeridos por el pipeline.
    
    1. y (Objetivo)
    2. t_series (Tiempo)
    3. passthrough_df (Datos que Script 1 ignora pero Script 2 necesita)
    4. pca_input_df (Datos que Script 1 comprimirá en PC*)
    """
    # O(M)
    logging.info("Identificando y separando columnas del pipeline...")

    # 1. Validar y extraer Objetivo
    if target_col not in df.columns:
        logging.error(f"Columna objetivo '{target_col}' no encontrada.")
        raise DataValidationError(f"Columna objetivo '{target_col}' no encontrada.")
    y = df[target_col]

    # 2. Validar y extraer Tiempo
    t_series = None
    if time_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[time_col]):
            t_series = df[time_col]
        else:
            logging.warning(f"'{time_col}' no es numérica. Tratada como passthrough.")

    # 3. Identificar columnas de características crudas para PCA
    pca_input_cols = [
        col for col in df.columns if col.startswith(raw_feature_prefix)
    ]
    if not pca_input_cols:
        logging.warning(f"No se encontraron columnas con prefijo "
                        f"'{raw_feature_prefix}'. El PCA no hará nada.")
        pca_input_df = pd.DataFrame(index=df.index)
    else:
        pca_input_df = df[pca_input_cols]

    # 4. Identificar todas las columnas 'passthrough'
    passthrough_explicit = set(
        col for sublist in passthrough_lists for col in sublist
    )
    cols_to_exclude = set(pca_input_cols) | {target_col, time_col}
    passthrough_all_cols = [
        col for col in df.columns if col not in cols_to_exclude
    ]
    
    missing_explicit = passthrough_explicit - set(df.columns)
    if missing_explicit:
        logging.warning(f"Faltan columnas 'Passthrough' esperadas: "
                        f"{missing_explicit}")

    passthrough_df = df[passthrough_all_cols]

    logging.info(f"Identificación completa:")
    logging.info(f"  {len(pca_input_cols)} columnas para PCA (prefijo "
                 f"'{raw_feature_prefix}')")
    logging.info(f"  {len(passthrough_all_cols)} columnas 'Passthrough' "
                 f"(para Script 2)")
    
    return y, t_series, passthrough_df, pca_input_df

def analyze_temporal_frequency(time_series: Optional[pd.Series]) -> None:
    """Analiza y reporta la frecuencia temporal de los datos."""
    # O(N log N)
    if time_series is None or time_series.empty:
        logging.debug("No hay serie temporal que analizar.")
        return
    logging.info("Iniciando análisis de frecuencia temporal...")
    try:
        if not time_series.is_monotonic_increasing:
            time_series = time_series.sort_values()
        deltas_t = time_series.diff().dropna()
        if deltas_t.empty:
            logging.warning("No se pudieron calcular deltas de tiempo.")
            return
        common_freq = deltas_t.mode()
        if not common_freq.empty:
            logging.info(f"Análisis Temporal: Frecuencia más común = "
                         f"{common_freq.iloc[0]} segundos.")
        logging.info(f"  Estadísticas de Frecuencia (seg): "
                     f"Media={deltas_t.mean():.2f}, "
                     f"Mediana={deltas_t.median():.2f}")
    except Exception as e:
        logging.warning(f"No se pudo completar el análisis temporal: {e}")

def scale_features(X: pd.DataFrame) -> np.ndarray:
    """Estandariza las características (media=0, varianza=1) e imputa NaNs."""
    # O(N*P_raw)
    logging.info(f"Estandarizando {X.shape[1]} características crudas...")
    if X.empty:
        logging.warning("DataFrame de características para escalar está vacío.")
        return np.array([[]] * len(X))
    if X.isnull().values.any():
        logging.warning(f"NaNs detectados ({X.isnull().sum().sum()} celdas). "
                        f"Imputando con la media de la columna.")
        X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Escalado completado.")
    return X_scaled

def fit_pca_optimized(X_scaled: np.ndarray, 
                        strategy: str, 
                        n_estimate: int) -> PCA:
    """
    Ajusta un modelo PCA usando la estrategia de optimización seleccionada.
    """
    # 'randomized': O(N*k_est^2) | 'full': O(N*P^2)
    if X_scaled.shape[1] == 0:
        logging.warning("fit_pca: No hay datos para ajustar PCA. "
                        "Devolviendo modelo vacío.")
        return PCA()
    logging.info(f"Ajustando modelo PCA (Estrategia='{strategy}')...")
    if strategy == 'randomized':
        n_comp_max = min(X_scaled.shape)
        n_comp_est = min(n_estimate, n_comp_max)
        if n_comp_est < n_estimate:
            logging.warning(f"PCA_N_COMPONENTS_ESTIMATE ({n_estimate}) > "
                            f"min(N,P) ({n_comp_max}). Usando n_components={n_comp_est}.")
        logging.info(f"Usando SVD Aleatorizado con n_components={n_comp_est}")
        pca_model = PCA(n_components=n_comp_est, 
                        svd_solver='randomized', 
                        random_state=42)
    elif strategy == 'full':
        logging.info("Usando SVD Completo (n_components=None).")
        pca_model = PCA(n_components=None, svd_solver='full', random_state=42)
    else:
        raise ValueError("PCA_STRATEGY debe ser 'randomized' o 'full'.")
    pca_model.fit(X_scaled)
    logging.info(f"PCA ajustado. {pca_model.n_components_} componentes "
                 f"analizadas.")
    return pca_model

def determine_optimal_components(pca_model: PCA, strategy: str, 
                                 variance_threshold: float = 0.999999
                                 ) -> Tuple[int, int, Optional[int]]:
    """
    Determina 'k' óptimo usando Codo y (opcionalmente) Varianza.
    """
    # O(P) o O(k_est)
    if not hasattr(pca_model, 'explained_variance_'):
        logging.warning("determine_optimal_components: Modelo PCA no ajustado.")
        return 0, 0, None
    logging.info("Determinando 'k' óptimo para PCA...")
    eigenvalues = pca_model.explained_variance_
    x_axis = range(1, len(eigenvalues) + 1)
    k_elbow, k_variance = None, None
    try:
        kn = KneeLocator(x_axis, eigenvalues, 
                         curve='convex', direction='decreasing')
        k_elbow = kn.knee
        if k_elbow is None:
            logging.warning("  [Codo] Kneedle no encontró codo. Usando k=1.")
            k_elbow = 1
        else:
            logging.info(f"  [Codo] Kneedle detectó codo en k = {k_elbow}.")
    except Exception as e:
        logging.error(f"  [Codo] Error en Kneedle: {e}. Usando k=1.")
        k_elbow = 1
    if strategy == 'full':
        cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
        k_variance = np.searchsorted(cumulative_variance, 
                                     variance_threshold) + 1
        logging.info(f"  [Varianza] k = {k_variance} retiene >= "
                     f"{variance_threshold * 100:.6f}% de varianza.")
        k_optimal = k_variance
        logging.info(f"-> 'k' Óptimo (Estrategia='full') seleccionado: {k_optimal}")
    else: # 'randomized'
        k_optimal = k_elbow
        logging.info(f"-> 'k' Óptimo (Estrategia='randomized') seleccionado: {k_optimal}")
    return k_optimal, k_elbow, k_variance

def plot_pca_diagnostics(pca_model: PCA, k_optimal: int, k_elbow: int, 
                         k_variance: Optional[int], output_file: str) -> None:
    """Genera y guarda un gráfico de diagnóstico PCA."""
    # O(P) o O(k_est)
    if not hasattr(pca_model, 'explained_variance_') or k_optimal == 0:
        logging.warning("plot_pca_diagnostics: Omitiendo gráfico, "
                        "PCA no ajustado o sin componentes.")
        return
    logging.info(f"Generando gráfico de diagnóstico PCA en: {output_file}")
    eigenvalues = pca_model.explained_variance_
    x_axis = range(1, len(eigenvalues) + 1)
    try:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        color = 'tab:blue'
        ax1.set_xlabel('Número de Componentes (k)', fontsize=12)
        ax1.set_ylabel('Eigenvalue (Varianza Explicada)', color=color, fontsize=12)
        ax1.plot(x_axis, eigenvalues, 'o-', color=color, 
                 label='Eigenvalues (Scree Plot)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axvline(k_elbow, color='red', linestyle='--', 
                    label=f'Método del Codo (k={k_elbow})')
        ax1.set_title('Diagnóstico PCA (Pre-procesamiento)', fontsize=16)
        if k_variance is not None:
            threshold = 0.999999
            cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Varianza Acumulada', color=color, fontsize=12)
            ax2.plot(x_axis, cumulative_variance, 's-', color=color, alpha=0.7, 
                     label='Varianza Acumulada')
            ax2.axhline(threshold, color='purple', linestyle=':', 
                        label=f'{threshold*100:.6f}% Umbral')
            ax2.axvline(k_variance, color='purple', linestyle=':', 
                        label=f'k Varianza (k={k_variance})')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 1.05)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='center right')
        else:
            ax1.legend(loc='upper right')
        fig.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        logging.info("Gráfico de diagnóstico guardado.")
    except Exception as e:
        logging.error(f"No se pudo generar el gráfico de diagnóstico PCA: {e}")

def transform_to_optimal_pca(X_scaled: np.ndarray, 
                             pca_model: PCA, 
                             k_optimal: int, 
                             index: pd.Index) -> pd.DataFrame:
    """
    Transforma los datos escalados a las 'k' componentes óptimas (PC*).
    """
    # O(N*P_raw*k) o O(N*k_est*k)
    if k_optimal == 0 or X_scaled.shape[1] == 0:
        logging.warning("transform_to_optimal_pca: No hay componentes óptimas "
                        "o datos de entrada. Devolviendo DataFrame vacío.")
        return pd.DataFrame(index=index)
    logging.info(f"Transformando datos a {k_optimal} componentes principales...")
    W_k = pca_model.components_[:k_optimal, :].T
    Zk_array = X_scaled.dot(W_k)
    # ¡¡CRÍTICO!!: Nombrar 'PC*' para que 'Script 2' las reconozca
    pca_cols = [f'PC{i+1}' for i in range(k_optimal)]
    Zk_df = pd.DataFrame(Zk_array, columns=pca_cols, index=index)
    logging.info(f"Transformación PCA completada. Forma de Zk: {Zk_df.shape}")
    return Zk_df

def run_principal_component_regression(Zk_df: pd.DataFrame, 
                                      y: pd.Series
                                      ) -> Tuple[pd.Series, pd.Series, 
                                                 Dict[str, float]]:
    """
    Ejecuta PCR (y ~ Zk) y calcula el 'Error_Lineal'.
    """
    # O(N*k^2 + k^3)
    logging.info("Ejecutando Regresión Lineal Inicial (PCR sobre PC*)...")
    if Zk_df.empty:
        logging.warning("PCR omitida: No hay componentes principales (Zk).")
        error_lineal = pd.Series(np.nan, index=y.index, name='Error_Lineal')
        y_pred_lineal = pd.Series(np.nan, index=y.index, name='Lineal_Pred')
        metrics = {"RMSE_Lineal": np.nan, "R2_Lineal": np.nan}
        return y_pred_lineal, error_lineal, metrics
    y_aligned = y.loc[Zk_df.index]
    if y_aligned.isnull().any():
        logging.warning(f"{y_aligned.isnull().sum()} NaNs en 'Objetivo'. "
                        f"Se eliminarán filas para la regresión.")
        valid_idx = y_aligned.dropna().index
        Zk_df = Zk_df.loc[valid_idx]
        y_aligned = y_aligned.loc[valid_idx]
    linear_model = LinearRegression()
    linear_model.fit(Zk_df, y_aligned)
    y_pred_lineal_array = linear_model.predict(Zk_df)
    y_pred_lineal = pd.Series(y_pred_lineal_array, 
                              index=Zk_df.index, name='Lineal_Pred')
    error_lineal = y_aligned - y_pred_lineal
    error_lineal.name = 'Error_Lineal'
    y_pred_lineal = y_pred_lineal.reindex(y.index)
    error_lineal = error_lineal.reindex(y.index)
    rmse = np.sqrt(mean_squared_error(y_aligned, y_pred_lineal_array))
    r2 = r2_score(y_aligned, y_pred_lineal_array)
    metrics = {"RMSE_Lineal": rmse, "R2_Lineal": r2}
    logging.info(f"  Resultado Regresión Lineal: RMSE={rmse:.6f}, "
                 f"R2={r2 * 100:.4f}%")
    return y_pred_lineal, error_lineal, metrics

def calculate_static_weighting_matrix(Zk_df: pd.DataFrame, 
                                      y: pd.Series, 
                                      error_lineal: pd.Series
                                      ) -> pd.DataFrame:
    """Calcula la matriz de correlación estática (ponderación)."""
    # O(N*k^2)
    if Zk_df.empty:
        logging.warning("Cálculo de correlación estática omitido (No hay PC*).")
        return pd.DataFrame(columns=[TARGET_COLUMN, 'Error_Lineal'])
    logging.info("Calculando Matriz de Correlación Estática (PC* vs Error)...")
    analysis_df = pd.concat([Zk_df, 
                             y.loc[Zk_df.index], 
                             error_lineal.loc[Zk_df.index]], axis=1)
    corr_matrix = analysis_df.corr(method='pearson')
    weighting_matrix = corr_matrix.loc[Zk_df.columns, 
                                       [y.name, error_lineal.name]]
    logging.info("Matriz de correlación estática calculada.")
    return weighting_matrix

# --- OPTIMIZACIÓN DE VECTORIZACIÓN ROBUSTA (v4.0) ---

def _vectorized_rolling_sum(A: np.ndarray, W: int) -> np.ndarray:
    """Calcula la suma rodante vectorizada (rolling sum) [O(N*k)]."""
    cumsum = np.cumsum(A, axis=0)
    rolling_sum = cumsum[W-1:]
    rolling_sum[W:] = rolling_sum[W:] - cumsum[:-W]
    padding_shape = (W - 1,) + A.shape[1:]
    padding = np.full(padding_shape, np.nan)
    return np.concatenate([padding, rolling_sum], axis=0)

def vectorized_rolling_correlation(X_df: pd.DataFrame, 
                                   y_s: pd.Series, 
                                   window: int) -> pd.DataFrame:
    """
    Calcula la correlación rodante (N,k) vs (N,) de forma vectorizada.
    Complejidad: O(N*k)
    """
    W = window
    X = X_df.values
    y = y_s.values.reshape(-1, 1)
    sum_X = _vectorized_rolling_sum(X, W)
    sum_Y = _vectorized_rolling_sum(y, W)
    sum_X2 = _vectorized_rolling_sum(X**2, W)
    sum_Y2 = _vectorized_rolling_sum(y**2, W)
    sum_XY = _vectorized_rolling_sum(X * y, W)
    mean_X = sum_X / W
    mean_Y = sum_Y / W
    cov_XY = (sum_XY / W) - (mean_X * mean_Y)
    var_X = (sum_X2 / W) - (mean_X**2)
    var_Y = (sum_Y2 / W) - (mean_Y**2)
    epsilon = 1e-10
    std_X = np.sqrt(var_X + epsilon)
    std_Y = np.sqrt(var_Y + epsilon)
    rho = cov_XY / (std_X * std_Y)
    corr_df = pd.DataFrame(rho, index=X_df.index, columns=X_df.columns)
    return corr_df

def calculate_dynamic_correlations(Zk_df: pd.DataFrame, 
                                   y: pd.Series, 
                                   error_lineal: pd.Series, 
                                   window_size: int
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orquestador del análisis de correlación dinámica (Vectorizado O(N*k))."""
    if Zk_df.empty:
        logging.warning("Cálculo de correlación dinámica omitido (No hay PC*).")
        return pd.DataFrame(), pd.DataFrame()
    logging.info(f"Calculando Correlación Dinámica (Vectorizada, W={window_size})...")
    if window_size > len(Zk_df):
        logging.warning(f"Ventana ({window_size}) > Datos ({len(Zk_df)}). "
                        f"Omitiendo correlación dinámica.")
        return pd.DataFrame(), pd.DataFrame()
    rolling_corr_target = vectorized_rolling_correlation(
        Zk_df, y.loc[Zk_df.index], window_size
    )
    rolling_corr_target.columns = [
        f'{col}_corr_{y.name}' for col in Zk_df.columns
    ]
    rolling_corr_error = vectorized_rolling_correlation(
        Zk_df, error_lineal.loc[Zk_df.index].fillna(0),
        window_size
    )
    rolling_corr_error.columns = [
        f'{col}_corr_{error_lineal.name}' for col in Zk_df.columns
    ]
    logging.info("Análisis de correlación dinámica vectorizado completado.")
    return rolling_corr_target, rolling_corr_error

def save_artifacts(output_file_path: str,
                   Zk_df: pd.DataFrame, 
                   y: pd.Series, 
                   error_lineal: pd.Series,
                   passthrough_df: pd.DataFrame,
                   analysis_artifacts: Dict[str, pd.DataFrame],
                   analysis_filepaths: Dict[str, str]) -> None:
    """
    Guarda el archivo de salida principal (para Script 2) y los
    artefactos de análisis (para este script).
    """
    # O(N*M) - Limitado por I/O
    logging.info("Guardando artefactos de salida...")
    
    # --- 1. Crear y Guardar el Archivo Principal (para Script 2) ---
    logging.info(f"Ensamblando archivo principal para 'Script 2'...")
    
    # Concatenar todas las partes que Script 2 necesita
    final_output_df = pd.concat([
        passthrough_df,
        y,
        Zk_df,          # Columnas PC*
        error_lineal    # Columna Error_Lineal (se vuelve 'other_var')
    ], axis=1)

    try:
        # ¡¡CRÍTICO v4.0!!: Guardar como .csv con índice (Date)
        # Script 2 usa: index_col='Date'
        final_output_df.to_csv(output_file_path, 
                               index=True,
                               index_label='Date') # Asegura nombre del índice
        
        logging.info(f"Archivo de salida principal (para Script 2) "
                     f"guardado en: {output_file_path}")
        logging.info(f"  Forma del archivo: {final_output_df.shape}")
    except Exception as e:
        logging.error(f"No se pudo guardar el archivo principal en "
                      f"{output_file_path}: {e}")
        
    del final_output_df # Liberar memoria
    gc.collect()

    # --- 2. Guardar Artefactos de Análisis (en Parquet por eficiencia) ---
    logging.info("Guardando artefactos de análisis (correlaciones)...")
    for key, df in analysis_artifacts.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            logging.warning(f"Artefacto '{key}' está vacío, se omite.")
            continue
        filepath = analysis_filepaths.get(key)
        if filepath:
            try:
                df.to_parquet(filepath, index=True, compression='snappy')
                logging.info(f"Artefacto '{key}' guardado en: {filepath}")
            except Exception as e:
                logging.error(f"No se pudo guardar '{key}' en {filepath}: {e}")

# SECCIÓN N: DEFINIR FUNCIÓN CONTROLADORA DEL PROCESO GLOBAL

def main_process_controller() -> None:
    """
    Función principal que orquesta todo el pipeline optimizado del Script 1.
    """
    setup_logging(OUTPUT_LOG_FILE)
    logger = logging.getLogger(__name__)

    try:
        logger.info("======================================================")
        logger.info(f"INICIO SCRIPT 1 (v4.0 Final: PCA={PCA_STRATEGY})")
        logger.info("======================================================")

        # 1. Carga (CSV)
        df = load_and_validate_data(INPUT_DATA_FILE)

        # 2. Identificación (Separación para Script 1 y Script 2)
        passthrough_lists = [
            PASSTHROUGH_COLS_CONSTANT, 
            PASSTHROUGH_COLS_TEMPORAL
        ]
        y, t_series, passthrough_df, pca_input_df = identify_columns_for_pipeline(
            df, 
            TARGET_COLUMN, 
            TIME_COLUMN,
            passthrough_lists,
            RAW_FEATURE_PREFIX
        )
        
        # 3. Análisis Temporal (sobre 'Unix' si existe)
        analyze_temporal_frequency(t_series)
        
        original_index = df.index
        del df
        gc.collect()

        # 4. Escalar (Solo columnas 'Raw_Feature_*')
        X_scaled = scale_features(pca_input_df)
        del pca_input_df
        gc.collect()

        # 5. Ajustar PCA
        pca_model = fit_pca_optimized(X_scaled, 
                                      PCA_STRATEGY, 
                                      PCA_N_COMPONENTS_ESTIMATE)
        
        # 6. Determinar 'k'
        k_optimal, k_elbow, k_variance = determine_optimal_components(
            pca_model, PCA_STRATEGY
        )
        
        # 7. Graficar
        plot_pca_diagnostics(
            pca_model, k_optimal, k_elbow, k_variance, 
            OUTPUT_PCA_DIAGNOSTICS_PLOT
        )
        
        # 8. Transformar (Crear 'PC1', 'PC2', ...)
        Zk_df = transform_to_optimal_pca(X_scaled, pca_model, 
                                         k_optimal, original_index)
        
        del X_scaled, pca_model
        gc.collect()

        # 9. PCR (y ~ Zk) para obtener Error_Lineal
        y_pred, error_lineal, metrics = run_principal_component_regression(Zk_df, y)
        logger.info(f"Métricas de Regresión Lineal (PCR): {metrics}")
        
        # 10. Correlación Estática
        static_corr_weights = calculate_static_weighting_matrix(Zk_df, y, 
                                                                error_lineal)
        
        if not static_corr_weights.empty:
            logger.info("\n--- Análisis de Ponderación Estática (Top 5 por Error Abs) ---")
            static_corr_sorted = static_corr_weights.reindex(
                static_corr_weights['Error_Lineal'].abs().sort_values(
                    ascending=False).index
            )
            logger.info(f"\n{static_corr_sorted.head().to_string()}")
        
        # 11. Correlación Dinámica (Vectorización O(N*k))
        dyn_corr_target, dyn_corr_error = calculate_dynamic_correlations(
            Zk_df, y, error_lineal, DYNAMIC_WINDOW_SIZE
        )

        # 12. Preparar y Guardar Artefactos (¡Salida CSV!)
        analysis_artifacts = {
            "static_correlation": static_corr_weights,
            "dynamic_corr_target": dyn_corr_target,
            "dynamic_corr_error": dyn_corr_error
        }
        analysis_filepaths = {
            "static_correlation": OUTPUT_STATIC_CORR_FILE,
            "dynamic_corr_target": OUTPUT_DYNAMIC_CORR_TARGET_FILE,
            "dynamic_corr_error": OUTPUT_DYNAMIC_CORR_ERROR_FILE
        }
        
        save_artifacts(
            output_file_path=OUTPUT_DATA_FILE,
            Zk_df=Zk_df,
            y=y,
            error_lineal=error_lineal,
            passthrough_df=passthrough_df,
            analysis_artifacts=analysis_artifacts,
            analysis_filepaths=analysis_filepaths
        )

        logger.info("======================================================")
        logger.info("FIN SCRIPT 1 (v4.0): PROCESO FINALIZADO")
        logger.info("======================================================")

    except DataValidationError as e:
        logging.critical(f"Error de validación de datos. Pipeline detenido: {e}",
                         exc_info=True)
    except MemoryError:
        logging.critical("Error de Memoria. Proceso detenido.", exc_info=True)
    except Exception as e:
        logging.critical(f"Error fatal inesperado: {e}", exc_info=True)
    finally:
        logging.shutdown()

# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    # --- Simulación de Datos Crudos (Si es necesario) ---
    # Genera un 'raw_data_input.csv' si no existe.
    
    if not os.path.exists(INPUT_DATA_FILE):
        logging.info(f"No se encontró '{INPUT_DATA_FILE}'. "
                     f"Generando datos crudos de demostración...")
        n_samples = 1000
        n_raw_features = 150 # Características para PCA
        
        df_demo = pd.DataFrame(
            index=pd.date_range(start='2023-01-01', 
                                periods=n_samples, freq='H', name='Date')
        )
        
        # 1. Columnas Passthrough (Constantes)
        for col in PASSTHROUGH_COLS_CONSTANT:
            df_demo[col] = np.random.rand() * 10
        # 2. Columnas Passthrough (Temporales)
        for col in PASSTHROUGH_COLS_TEMPORAL:
            df_demo[col] = np.random.rand(n_samples)
        # 3. Columnas 'other_vars' (también passthrough)
        df_demo['Other_Var_1'] = np.random.rand(n_samples)
        df_demo['Other_Var_2'] = np.random.rand(n_samples)
        
        # 4. Columnas para PCA (Raw Features)
        raw_cols = {
            f'{RAW_FEATURE_PREFIX}{i+1}': np.random.randn(n_samples) 
            for i in range(n_raw_features)
        }
        df_demo = pd.concat([df_demo, pd.DataFrame(raw_cols, 
                                                   index=df_demo.index)], 
                            axis=1)
        
        # 5. Target y Time
        df_demo[TIME_COLUMN] = df_demo.index.astype(int) // 10**9
        df_demo[TARGET_COLUMN] = (
            df_demo[f'{RAW_FEATURE_PREFIX}1'] * 0.5 + 
            df_demo[f'{RAW_FEATURE_PREFIX}2'] * -0.3 + 
            np.sin(df_demo[f'{RAW_FEATURE_PREFIX}3'] * 5) +
            df_demo['Other_Var_1'] * 0.2 +
            np.random.randn(n_samples) * 0.1
        )
        # Guardar como CSV con el índice 'Date'
        df_demo.to_csv(INPUT_DATA_FILE, index=True, index_label='Date')
        logging.info(f"Datos de demostración guardados en '{INPUT_DATA_FILE}'")

    # --- Ejecutar el Proceso Controlador ---
    main_process_controller()
