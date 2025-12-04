"""
Módulo de optimización de hiperparámetros para modelos LSTM.

Este módulo realiza búsqueda sistemática de los mejores hiperparámetros
para el modelo de series temporales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from .data_loader import TimeSeriesDataLoader
from .model import LSTMModel, create_lstm_model, create_bidirectional_lstm
from .train import train_model
from .evaluate import evaluate_model, calculate_metrics


class HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para modelos LSTM de series temporales.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Inicializa el optimizador.
        
        Args:
            data: DataFrame con los datos
            target_column: Nombre de la columna objetivo
        """
        self.data = data
        self.target_column = target_column
        self.results = []
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')
        
    def run_experiment(
        self,
        sequence_length: int,
        lstm_units: List[int],
        dropout_rate: float,
        learning_rate: float,
        batch_size: int,
        use_bidirectional: bool = False,
        epochs: int = 100,
        test_size: float = 0.2,
        verbose: int = 0
    ) -> Dict:
        """
        Ejecuta un experimento con los hiperparámetros dados.
        """
        # Preparar datos
        loader = TimeSeriesDataLoader(sequence_length=sequence_length)
        loader.load_from_dataframe(self.data.copy(), target_column=self.target_column)
        X_train, X_test, y_train, y_test = loader.prepare_data(test_size=test_size)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Crear modelo
        model = LSTMModel(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            use_bidirectional=use_bidirectional,
            dense_units=[lstm_units[-1] // 2] if lstm_units else [25]
        )
        
        # Entrenar
        train_result = train_model(
            model, X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=15,
            save_best=False,
            verbose=verbose
        )
        
        # Evaluar
        eval_result = evaluate_model(model, X_test, y_test, loader.scaler, verbose=False)
        
        return {
            'model': model,
            'loader': loader,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_result': train_result,
            'eval_result': eval_result,
            'metrics': eval_result['metrics']
        }
    
    def grid_search(
        self,
        param_grid: Dict,
        epochs: int = 100,
        test_size: float = 0.2,
        verbose: int = 0
    ) -> pd.DataFrame:
        """
        Realiza búsqueda en grid de hiperparámetros.
        
        Args:
            param_grid: Diccionario con listas de valores para cada hiperparámetro
            epochs: Número máximo de épocas
            test_size: Proporción de datos para test
            verbose: Nivel de verbosidad
            
        Returns:
            DataFrame con resultados ordenados por RMSE
        """
        # Generar todas las combinaciones
        keys = param_grid.keys()
        combinations = list(product(*param_grid.values()))
        
        print(f"\n{'='*60}")
        print(f"BÚSQUEDA DE HIPERPARÁMETROS")
        print(f"{'='*60}")
        print(f"Total de combinaciones: {len(combinations)}")
        print(f"{'='*60}\n")
        
        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            
            print(f"\n[{i+1}/{len(combinations)}] Probando: {params}")
            
            try:
                result = self.run_experiment(
                    sequence_length=params.get('sequence_length', 60),
                    lstm_units=params.get('lstm_units', [50, 50]),
                    dropout_rate=params.get('dropout_rate', 0.2),
                    learning_rate=params.get('learning_rate', 0.001),
                    batch_size=params.get('batch_size', 32),
                    use_bidirectional=params.get('use_bidirectional', False),
                    epochs=epochs,
                    test_size=test_size,
                    verbose=verbose
                )
                
                metrics = result['metrics']
                
                # Guardar resultado
                experiment = {
                    **params,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2'],
                    'epochs_trained': result['train_result']['epochs_trained'],
                    'best_val_loss': result['train_result']['best_val_loss']
                }
                self.results.append(experiment)
                
                print(f"  → RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")
                
                # Actualizar mejor modelo
                if metrics['RMSE'] < self.best_score:
                    self.best_score = metrics['RMSE']
                    self.best_model = result['model']
                    self.best_params = params
                    self.best_result = result
                    print(f"  ★ Nuevo mejor modelo!")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('RMSE')
        
        print(f"\n{'='*60}")
        print("MEJORES CONFIGURACIONES")
        print(f"{'='*60}")
        print(results_df.head(10).to_string(index=False))
        print(f"\n{'='*60}")
        print(f"MEJOR MODELO:")
        print(f"  Parámetros: {self.best_params}")
        print(f"  RMSE: {self.best_score:.4f}")
        print(f"{'='*60}")
        
        return results_df
    
    def quick_search(self, epochs: int = 50) -> pd.DataFrame:
        """
        Búsqueda rápida con configuraciones predefinidas optimizadas.
        """
        param_grid = {
            'sequence_length': [30, 60],
            'lstm_units': [[32], [64, 32], [50, 50]],
            'dropout_rate': [0.2, 0.3],
            'learning_rate': [0.001],
            'batch_size': [32],
            'use_bidirectional': [False]
        }
        
        return self.grid_search(param_grid, epochs=epochs, verbose=0)
    
    def full_search(self, epochs: int = 100) -> pd.DataFrame:
        """
        Búsqueda completa con muchas configuraciones.
        """
        param_grid = {
            'sequence_length': [30, 60, 90],
            'lstm_units': [[32], [64], [64, 32], [50, 50], [100, 50]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32, 64],
            'use_bidirectional': [False, True]
        }
        
        return self.grid_search(param_grid, epochs=epochs, verbose=0)


def analyze_data_characteristics(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Analiza características de los datos para recomendar configuraciones.
    """
    data = df[target_col]
    
    analysis = {
        'n_samples': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min(),
        'cv': data.std() / data.mean() * 100,  # Coeficiente de variación
    }
    
    # Calcular autocorrelación
    from statsmodels.tsa.stattools import acf
    acf_values = acf(data.dropna(), nlags=min(365, len(data)//4))
    
    # Encontrar lag con mayor correlación (excluyendo lag 0)
    significant_lags = np.where(np.abs(acf_values[1:]) > 0.2)[0] + 1
    analysis['significant_lags'] = significant_lags[:10].tolist() if len(significant_lags) > 0 else []
    
    # Recomendar sequence_length basado en autocorrelación
    if len(significant_lags) > 0:
        recommended_seq = min(max(significant_lags[:5]), 90)
    else:
        recommended_seq = 60
    analysis['recommended_sequence_length'] = int(recommended_seq)
    
    # Test de estacionariedad
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(data.dropna())
    analysis['is_stationary'] = adf_result[1] < 0.05
    analysis['adf_pvalue'] = adf_result[1]
    
    return analysis


def get_optimized_config(data_analysis: Dict) -> Dict:
    """
    Genera configuración optimizada basada en análisis de datos.
    """
    n_samples = data_analysis['n_samples']
    cv = data_analysis['cv']
    is_stationary = data_analysis['is_stationary']
    
    config = {}
    
    # Sequence length
    config['sequence_length'] = data_analysis['recommended_sequence_length']
    
    # Complejidad del modelo basada en cantidad de datos
    if n_samples < 500:
        config['lstm_units'] = [32]
        config['dense_units'] = []
    elif n_samples < 2000:
        config['lstm_units'] = [50, 50]
        config['dense_units'] = [25]
    else:
        config['lstm_units'] = [64, 64, 32]
        config['dense_units'] = [32]
    
    # Dropout basado en variabilidad
    if cv > 30:
        config['dropout_rate'] = 0.3
    else:
        config['dropout_rate'] = 0.2
    
    # Learning rate
    config['learning_rate'] = 0.001
    
    # Batch size basado en datos
    if n_samples < 1000:
        config['batch_size'] = 16
    else:
        config['batch_size'] = 32
    
    return config



