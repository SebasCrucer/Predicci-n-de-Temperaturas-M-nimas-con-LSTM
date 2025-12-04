"""
Módulo de evaluación para modelos LSTM de series temporales.

Este módulo proporciona:
- Cálculo de métricas de evaluación (MAE, RMSE, MAPE, R²)
- Visualización de predicciones vs valores reales
- Análisis de errores
- Reportes de evaluación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler

from .model import LSTMModel


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para predicciones.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas calculadas
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # MAE - Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # MSE - Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # MAPE - Mean Absolute Percentage Error
    # Evitar división por cero
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # R² - Coeficiente de determinación
    r2 = r2_score(y_true, y_pred)
    
    # Error máximo
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Error medio
    mean_error = np.mean(y_pred - y_true)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Max_Error': max_error,
        'Mean_Error': mean_error
    }


def evaluate_model(
    model: LSTMModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
    verbose: bool = True
) -> Dict:
    """
    Evalúa el modelo y calcula métricas.
    
    Args:
        model: Modelo entrenado (LSTMModel o Keras model)
        X_test: Datos de prueba
        y_test: Valores objetivo de prueba
        scaler: Escalador para invertir normalización
        verbose: Si mostrar resultados
        
    Returns:
        Diccionario con métricas y predicciones
    """
    # Obtener modelo Keras
    keras_model = model.model if isinstance(model, LSTMModel) else model
    
    # Realizar predicciones
    y_pred = keras_model.predict(X_test, verbose=0)
    
    # Invertir normalización si se proporciona scaler
    if scaler is not None:
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_test_original = y_test.flatten()
        y_pred_original = y_pred.flatten()
    
    # Calcular métricas
    metrics = calculate_metrics(y_test_original, y_pred_original)
    
    if verbose:
        print_metrics_report(metrics)
    
    return {
        'metrics': metrics,
        'y_true': y_test_original,
        'y_pred': y_pred_original,
        'y_true_scaled': y_test.flatten(),
        'y_pred_scaled': y_pred.flatten()
    }


def print_metrics_report(metrics: Dict[str, float]):
    """
    Imprime un reporte formateado de métricas.
    
    Args:
        metrics: Diccionario con métricas
    """
    print("\n" + "=" * 50)
    print("         MÉTRICAS DE EVALUACIÓN")
    print("=" * 50)
    print(f"  MAE  (Error Absoluto Medio):      {metrics['MAE']:.4f}")
    print(f"  RMSE (Raíz del Error Cuadrático): {metrics['RMSE']:.4f}")
    print(f"  MAPE (Error Porcentual):          {metrics['MAPE']:.2f}%")
    print(f"  R²   (Coef. Determinación):       {metrics['R2']:.4f}")
    print("-" * 50)
    print(f"  MSE  (Error Cuadrático Medio):    {metrics['MSE']:.4f}")
    print(f"  Error Máximo:                     {metrics['Max_Error']:.4f}")
    print(f"  Error Medio (Sesgo):              {metrics['Mean_Error']:.4f}")
    print("=" * 50)
    
    # Interpretación
    print("\n📊 INTERPRETACIÓN:")
    if metrics['MAPE'] < 10:
        print("  ✓ MAPE < 10%: Predicciones muy precisas")
    elif metrics['MAPE'] < 20:
        print("  ○ MAPE 10-20%: Predicciones buenas")
    else:
        print("  ✗ MAPE > 20%: Predicciones con margen de mejora")
    
    if metrics['R2'] > 0.9:
        print("  ✓ R² > 0.9: Excelente ajuste del modelo")
    elif metrics['R2'] > 0.7:
        print("  ○ R² 0.7-0.9: Buen ajuste del modelo")
    else:
        print("  ✗ R² < 0.7: Ajuste del modelo mejorable")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Predicciones vs Valores Reales",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        dates: Índice de fechas (opcional)
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la imagen
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        x_axis = dates[:len(y_true)]
    else:
        x_axis = range(len(y_true))
    
    ax.plot(x_axis, y_true, 'b-', label='Valores Reales', linewidth=2, alpha=0.8)
    ax.plot(x_axis, y_pred, 'r--', label='Predicciones', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Fecha' if dates is not None else 'Índice', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Añadir área de error
    ax.fill_between(x_axis, y_true, y_pred, alpha=0.2, color='gray', label='Error')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Gráfico de dispersión: predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la imagen
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, c='#2E86AB', edgecolors='white', linewidth=0.5)
    
    # Línea de predicción perfecta
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    # Calcular R²
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Predicciones', fontsize=12)
    ax.set_title(f'Predicciones vs Valores Reales\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hacer cuadrado el gráfico
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la distribución de errores.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la imagen
        
    Returns:
        Figura de matplotlib
    """
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histograma de errores
    axes[0].hist(errors, bins=50, edgecolor='white', color='#2E86AB', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(errors), color='orange', linestyle='--', linewidth=2, 
                    label=f'Media: {np.mean(errors):.4f}')
    axes[0].set_xlabel('Error (Real - Predicho)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Errores')
    axes[0].legend()
    
    # Errores a lo largo del tiempo
    axes[1].plot(errors, color='#2E86AB', alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].fill_between(range(len(errors)), errors, 0, alpha=0.3, color='#2E86AB')
    axes[1].set_xlabel('Índice')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Errores en el Tiempo')
    
    # Errores absolutos acumulados
    cumulative_mae = np.cumsum(np.abs(errors)) / np.arange(1, len(errors) + 1)
    axes[2].plot(cumulative_mae, color='#E94F37', linewidth=2)
    axes[2].set_xlabel('Índice')
    axes[2].set_ylabel('MAE Acumulado')
    axes[2].set_title('MAE Acumulado')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Errores de Predicción', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def full_evaluation_report(
    model: LSTMModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    output_dir: str = 'outputs',
    model_name: str = 'LSTM'
) -> Dict:
    """
    Genera un reporte de evaluación completo con todas las visualizaciones.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Valores objetivo de prueba
        scaler: Escalador para invertir normalización
        dates: Índice de fechas
        output_dir: Directorio de salida
        model_name: Nombre del modelo para títulos
        
    Returns:
        Diccionario con todos los resultados
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"REPORTE DE EVALUACIÓN: {model_name}")
    print("=" * 60)
    
    # Evaluar modelo
    results = evaluate_model(model, X_test, y_test, scaler, verbose=True)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Generar visualizaciones
    print("\n📊 GENERANDO VISUALIZACIONES...")
    
    plot_predictions(
        y_true, y_pred, dates,
        title=f'{model_name} - Predicciones vs Valores Reales',
        save_path=f'{output_dir}/predictions.png'
    )
    print("  ✓ Gráfico de predicciones guardado")
    
    plot_prediction_scatter(
        y_true, y_pred,
        save_path=f'{output_dir}/scatter_plot.png'
    )
    print("  ✓ Gráfico de dispersión guardado")
    
    plot_error_distribution(
        y_true, y_pred,
        save_path=f'{output_dir}/error_analysis.png'
    )
    print("  ✓ Análisis de errores guardado")
    
    # Guardar métricas en archivo
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
    print(f"  ✓ Métricas guardadas en: {output_dir}/metrics.csv")
    
    print("\n" + "=" * 60)
    print(f"Reporte completado. Archivos en: {output_dir}/")
    print("=" * 60)
    
    plt.close('all')
    
    return results


def compare_models(
    models: List[Tuple[str, LSTMModel]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[MinMaxScaler] = None
) -> pd.DataFrame:
    """
    Compara múltiples modelos y genera tabla comparativa.
    
    Args:
        models: Lista de tuplas (nombre, modelo)
        X_test: Datos de prueba
        y_test: Valores objetivo
        scaler: Escalador para invertir normalización
        
    Returns:
        DataFrame con comparación de métricas
    """
    results = []
    
    print("\nComparando modelos...")
    print("-" * 50)
    
    for name, model in models:
        eval_result = evaluate_model(model, X_test, y_test, scaler, verbose=False)
        metrics = eval_result['metrics']
        metrics['Model'] = name
        results.append(metrics)
        print(f"  {name}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
    
    df = pd.DataFrame(results)
    df = df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']]
    
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return df



