"""
Módulo de Series Temporales con LSTM

Este paquete contiene las herramientas para:
- Carga y preprocesamiento de datos
- Análisis exploratorio de series temporales
- Modelo LSTM para predicción
- Entrenamiento y evaluación
- Optimización de hiperparámetros
"""

from .data_loader import TimeSeriesDataLoader
from .model import LSTMModel, create_lstm_model
from .train import train_model
from .evaluate import evaluate_model, calculate_metrics
from .optimizer import HyperparameterOptimizer, analyze_data_characteristics, get_optimized_config

__all__ = [
    'TimeSeriesDataLoader',
    'LSTMModel',
    'create_lstm_model',
    'train_model',
    'evaluate_model',
    'calculate_metrics',
    'HyperparameterOptimizer',
    'analyze_data_characteristics',
    'get_optimized_config'
]

