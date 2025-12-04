"""
Módulo de entrenamiento para modelos LSTM de series temporales.

Este módulo proporciona:
- Funciones de entrenamiento con callbacks configurables
- Early stopping y model checkpoint
- Seguimiento del historial de entrenamiento
- Visualización del proceso de entrenamiento
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, 
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.models import Sequential
from typing import Optional, Dict, Tuple, List
import os
from datetime import datetime

from .model import LSTMModel


def train_model(
    model: LSTMModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.1,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 7,
    min_delta: float = 0.0001,
    save_best: bool = True,
    model_path: str = 'outputs/best_model.keras',
    verbose: int = 1
) -> Dict:
    """
    Entrena el modelo LSTM con callbacks configurados.
    
    Args:
        model: Instancia de LSTMModel o modelo Keras Sequential
        X_train: Datos de entrenamiento (secuencias)
        y_train: Valores objetivo de entrenamiento
        X_val: Datos de validación (opcional)
        y_val: Valores objetivo de validación (opcional)
        epochs: Número máximo de épocas
        batch_size: Tamaño del batch
        validation_split: Proporción de datos para validación (si X_val no se proporciona)
        early_stopping_patience: Épocas sin mejora antes de parar
        reduce_lr_patience: Épocas sin mejora antes de reducir learning rate
        min_delta: Mínimo cambio para considerar mejora
        save_best: Si guardar el mejor modelo
        model_path: Ruta para guardar el modelo
        verbose: Nivel de verbosidad (0, 1, o 2)
        
    Returns:
        Diccionario con historial de entrenamiento y métricas
    """
    # Obtener modelo Keras
    keras_model = model.model if isinstance(model, LSTMModel) else model
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else 'outputs', exist_ok=True)
    
    # Configurar callbacks
    callbacks = []
    
    # Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=verbose
    )
    callbacks.append(early_stop)
    
    # Reducir Learning Rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=verbose
    )
    callbacks.append(reduce_lr)
    
    # Model Checkpoint
    if save_best:
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=verbose
        )
        callbacks.append(checkpoint)
    
    # Preparar datos de validación
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        validation_split = 0.0
    
    # Entrenar modelo
    print("\n" + "=" * 60)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 60)
    print(f"Épocas máximas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Datos de entrenamiento: {X_train.shape}")
    if validation_data:
        print(f"Datos de validación: {X_val.shape}")
    else:
        print(f"Validation split: {validation_split:.0%}")
    print("=" * 60 + "\n")
    
    start_time = datetime.now()
    
    history = keras_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split if validation_data is None else 0.0,
        callbacks=callbacks,
        verbose=verbose
    )
    
    training_time = datetime.now() - start_time
    
    # Resumen de entrenamiento
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Tiempo de entrenamiento: {training_time}")
    print(f"Épocas ejecutadas: {len(history.history['loss'])}")
    print(f"Mejor loss de validación: {min(history.history['val_loss']):.6f}")
    print(f"Loss final de entrenamiento: {history.history['loss'][-1]:.6f}")
    if save_best:
        print(f"Mejor modelo guardado en: {model_path}")
    print("=" * 60)
    
    return {
        'history': history.history,
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': min(history.history['val_loss']),
        'final_train_loss': history.history['loss'][-1],
        'training_time': str(training_time),
        'model_path': model_path if save_best else None
    }


def plot_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza el historial de entrenamiento.
    
    Args:
        history: Diccionario con historial (o resultado de train_model)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la imagen
        
    Returns:
        Figura de matplotlib
    """
    # Extraer historial si viene de train_model
    if 'history' in history:
        history = history['history']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Gráfico de Loss
    axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Evolución del Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Marcar mejor época
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[0].scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    axes[0].annotate(f'Mejor: {best_val_loss:.4f}\nÉpoca {best_epoch}', 
                     xy=(best_epoch, best_val_loss),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='green')
    
    # Gráfico de MAE si está disponible
    if 'mae' in history:
        axes[1].plot(epochs, history['mae'], 'b-', label='Training MAE', linewidth=2)
        axes[1].plot(epochs, history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Evolución del MAE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        # Si no hay MAE, mostrar loss en escala logarítmica
        axes[1].semilogy(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[1].semilogy(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Loss (log scale)', fontsize=12)
        axes[1].set_title('Loss en Escala Logarítmica', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def cross_validate_timeseries(
    model_class,
    model_params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    **train_params
) -> Dict:
    """
    Realiza validación cruzada temporal (time series split).
    
    Args:
        model_class: Clase del modelo (LSTMModel)
        model_params: Parámetros para inicializar el modelo
        X: Datos de entrada completos
        y: Valores objetivo completos
        n_splits: Número de divisiones
        **train_params: Parámetros para train_model
        
    Returns:
        Diccionario con resultados de validación cruzada
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {
        'val_losses': [],
        'train_losses': [],
        'histories': []
    }
    
    print(f"\nValidación Cruzada Temporal ({n_splits} splits)")
    print("=" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Crear nuevo modelo para cada fold
        model = model_class(**model_params)
        
        # Entrenar
        train_result = train_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            save_best=False,
            verbose=0,
            **train_params
        )
        
        results['val_losses'].append(train_result['best_val_loss'])
        results['train_losses'].append(train_result['final_train_loss'])
        results['histories'].append(train_result['history'])
        
        print(f"  Val Loss: {train_result['best_val_loss']:.6f}")
    
    # Calcular estadísticas
    results['mean_val_loss'] = np.mean(results['val_losses'])
    results['std_val_loss'] = np.std(results['val_losses'])
    results['mean_train_loss'] = np.mean(results['train_losses'])
    
    print("\n" + "=" * 50)
    print("RESULTADOS DE VALIDACIÓN CRUZADA")
    print("=" * 50)
    print(f"Val Loss promedio: {results['mean_val_loss']:.6f} ± {results['std_val_loss']:.6f}")
    print(f"Train Loss promedio: {results['mean_train_loss']:.6f}")
    print("=" * 50)
    
    return results


class TrainingLogger:
    """
    Logger personalizado para seguimiento del entrenamiento.
    """
    
    def __init__(self, log_dir: str = 'outputs/logs'):
        """
        Inicializa el logger.
        
        Args:
            log_dir: Directorio para logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(
            log_dir, 
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    def log(self, message: str):
        """Escribe mensaje al log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(message)



