import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def lineal_regresion_PCA():
    # Carga de datos
    file_path = '/content/drive/input.csv'
    data = pd.read_csv(file_path, index_col='Date')

    # Configuración inicial
    n_PCA_temporal = 12
    n_PCA_others = 12

    # Escalado de variables constantes
    constant_vars = ['Frecuencia Grupal', 'C_Rel_Luz', 'Frecuencia Fundamental General', 'G_constant', 'Fundamental_Frequency', 'inc_t']
    data['Product_Constantes'] = data[constant_vars].prod(axis=1)
    scaler = StandardScaler()
    data['Scaled_Product_Constantes'] = scaler.fit_transform(data[['Product_Constantes']])

    # Aplicación de PCA a variables temporales
    temporal_vars = [
        'Minutes', 'Hour', 'Day', 'DayOfWeek', 'DayOfYear', 'Week', 'WeekOfYear', 'Month', 'Year',
        'sin_t', 'Model_dynamic_temporal', 'Model_static_temporal', 'Model_large_temporal',
        'Model_total_temporal', 'Model_combined', 'Model_combined_normalized', 'Seconds'
    ]
    scaler_temporal = StandardScaler()
    temporal_data_scaled = scaler_temporal.fit_transform(data[temporal_vars])
    pca_temporal = PCA(n_components=n_PCA_temporal)
    temporal_transformed = pca_temporal.fit_transform(temporal_data_scaled)

    # PCA para otras variables
    other_vars = [col for col in data.columns if col not in constant_vars + temporal_vars and ('PC' or 'Objetivo') not in col]
    scaler_others = StandardScaler()
    X_others_scaled = scaler_others.fit_transform(data[other_vars])
    pca_others = PCA(n_components=n_PCA_others)
    others_transformed = pca_others.fit_transform(X_others_scaled)
    # Crear DataFrame con las componentes de PCA
    pca_columns_temporal = [f'PCA_Temporal_{i}' for i in range(n_PCA_temporal)]
    pca_columns_others = [f'PCA_Others_{i}' for i in range(n_PCA_others)]
    pca_df = pd.DataFrame(np.hstack((temporal_transformed, others_transformed)), columns=pca_columns_temporal + pca_columns_others, index=data.index)

    # Añadir columnas al DataFrame original
    data = pd.concat([data, pca_df], axis=1)

    # Identificar las columnas PCA originales si existen
    pca_original_columns = [col for col in data.columns if 'PC' in col and col not in pca_columns_temporal + pca_columns_others]

    # Preparar el conjunto de datos para el modelo
    all_pca_components = data[pca_columns_temporal + pca_columns_others + pca_original_columns + ['Scaled_Product_Constantes']].values

    # División de los datos
    X_train, X_test, Y_train, Y_test = train_test_split(all_pca_components, data['Objetivo'], test_size=0.1, random_state=42)
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_regression', LinearRegression())
    ])

    # Validación cruzada para RMSE y R²
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_scores = cross_val_score(final_model, X_train, Y_train, cv=kf, scoring=make_scorer(rmse))
    r2_scores = cross_val_score(final_model, X_train, Y_train, cv=kf, scoring='r2')

    # Entrenar el modelo con todo el conjunto de entrenamiento
    final_model.fit(X_train, Y_train)
    Y_pred = final_model.predict(X_test)
    test_rmse = rmse(Y_test, Y_pred)
    test_r2 = final_model.score(X_test, Y_test)

    # Informes de rendimiento
    print("RMSE de validación cruzada:", rmse_scores)
    print("RMSE promedio de validación cruzada:", np.mean(rmse_scores))
    print(f"R² de validación cruzada:  {r2_scores*100}%")
    print(f"R² promedio de validación cruzada: {np.mean(r2_scores*100)}%")
    print("RMSE en el conjunto de prueba:", test_rmse)
    print(f"R² en el conjunto de prueba: {test_r2*100}%")

    # Predecir valores para todo el conjunto de datos
    data['Lineal_Pred'] = final_model.predict(all_pca_components)
    # Calcular errores
    data['Error'] = data['Objetivo'] - data['Lineal_Pred']

    # Visualización de residuales
    residuals = Y_test - Y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Residuales')
    plt.title('Análisis de Residuales')
    plt.show()

    # Visualización del error de predicción
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7, color='blue')
    plt.title('Distribución del Error de Predicción')
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')
    plt.show()

    # Gráfica de los resultados de la regresión lineal
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.3)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Valores Reales vs Predicciones')
    plt.grid(True)
    plt.show()

    def optimize_H(E, lambda_reg=0.01):
        """
        Optimiza la matriz de variaciones H para minimizar la suma de los cuadrados
        de los residuales ajustados más una penalización por la norma de Frobenius de H.

        Parámetros:
        - E (np.array): Matriz de residuales existente de tamaño n x 1.
        - lambda_reg (float): Parámetro de regularización que controla la magnitud de la penalización.

        Retorna:
        - H_optimal (np.array): Matriz optimizada de variaciones de tamaño n x 1.
        """
        n = E.shape[0]  # Número de observaciones

        # Función objetivo que se va a minimizar
        def objective(H):
            H = H.reshape(-1, 1)  # Asegurar que H es una matriz columna
            residual_adjusted = E + H
            penalty = lambda_reg * np.linalg.norm(H, 'fro')  # Penalización por norma de Frobenius
            return np.sum(residual_adjusted ** 2) + penalty

        # Aplanar la matriz H inicial (matriz de ceros)
        H_initial = np.zeros((n, 1))
        result = minimize(objective, H_initial.ravel(), method='L-BFGS-B')
        H_optimal = result.x.reshape(-1, 1)
        return H_optimal

    # Suponiendo que 'data' es el DataFrame y 'Error' es la columna con los residuales originales
    # Transformar la columna de error en una matriz de n x 1
    E = data['Error'].values.reshape(-1, 1)

    # Aplicar la función de optimización para obtener H óptima
    optimized_H = optimize_H(E)

    # Ajustar los residuales en el DataFrame
    data['Adjusted_Error'] = data['Error'] + optimized_H.flatten()

    # Visualización de residuales ajustados
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Objetivo'], data['Adjusted_Error'])
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Residuales Ajustados')
    plt.title('Análisis de Residuales Ajustados')
    plt.show()

    # Gráfica de los resultados ajustados
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Objetivo'], data['Objetivo'] - data['Adjusted_Error'], alpha=0.3)
    plt.plot([data['Objetivo'].min(), data['Objetivo'].max()], [data['Objetivo'].min(), data['Objetivo'].max()], 'k--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones Ajustadas')
    plt.title('Valores Reales vs Predicciones Ajustadas')
    plt.grid(True)
    plt.show()
    data['Adjusted_Pred'] = data['Objetivo'] - data['Adjusted_Error']


    # Cálculo de métricas para las predicciones ajustadas
    adjusted_rmse = np.sqrt(mean_squared_error(data['Objetivo'], data['Adjusted_Pred']))
    adjusted_r2 = r2_score(data['Objetivo'], data['Adjusted_Pred'])

    print(f"RMSE Ajustado: {adjusted_rmse}")
    print(f"R² Ajustado: {adjusted_r2*100}%")

    # Visualización de residuales ajustados
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Objetivo'], data['Adjusted_Error'])
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Residuales Ajustados')
    plt.title('Análisis de Residuales Ajustados')
    plt.show()

    # Gráfica de los resultados ajustados
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Objetivo'], data['Adjusted_Pred'], alpha=0.3)
    plt.plot([data['Objetivo'].min(), data['Objetivo'].max()], [data['Objetivo'].min(), data['Objetivo'].max()], 'k--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones Ajustadas')
    plt.title('Valores Reales vs Predicciones Ajustadas')
    plt.grid(True)
    plt.show()

    # Preparar el DataFrame para guardar
    output_columns = ['Unix', 'Objetivo'] + [f'PCA_Temporal_{i}' for i in range(n_PCA_temporal)] + [f'PCA_Others_{i}' for i in range(n_PCA_others)] + pca_original_columns + ['Scaled_Product_Constantes', 'Lineal_Pred', 'Error', 'Adjusted_Pred', 'Adjusted_Error']
    final_data = data[output_columns]

    # Guardar los resultados en un archivo CSV
    output_file_path = '/content/drive/ubicacion_datos.csv'
    final_data.to_csv(output_file_path, index_label='Date')

    print("Datos guardados en:", output_file_path)


if __name__ == "__main__":
    lineal_regresion_PCA()


