"""
Módulo de modelo LSTM para series temporales.

Este módulo proporciona:
- Arquitectura de modelo LSTM configurable
- Funciones de creación de modelos
- Configuraciones predefinidas para diferentes casos de uso
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, 
    Bidirectional, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from typing import Tuple, Optional, List, Dict


class LSTMModel:
    """
    Clase para crear y configurar modelos LSTM para series temporales.
    
    Attributes:
        model (Sequential): Modelo Keras compilado
        input_shape (Tuple): Forma de entrada (time_steps, features)
        config (Dict): Configuración del modelo
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        dense_units: List[int] = [25],
        output_units: int = 1,
        learning_rate: float = 0.001,
        use_bidirectional: bool = False,
        use_batch_norm: bool = False,
        l2_reg: float = 0.0
    ):
        """
        Inicializa y construye el modelo LSTM.
        
        Args:
            input_shape: Tupla (time_steps, features) - forma de los datos de entrada
            lstm_units: Lista con unidades para cada capa LSTM
            dropout_rate: Tasa de dropout (0-1)
            dense_units: Lista con unidades para capas Dense antes de la salida
            output_units: Número de unidades de salida (1 para predicción univariada)
            learning_rate: Tasa de aprendizaje para el optimizador
            use_bidirectional: Si usar LSTM bidireccional
            use_batch_norm: Si usar normalización por lotes
            l2_reg: Regularización L2
        """
        self.input_shape = input_shape
        self.config = {
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'dense_units': dense_units,
            'output_units': output_units,
            'learning_rate': learning_rate,
            'use_bidirectional': use_bidirectional,
            'use_batch_norm': use_batch_norm,
            'l2_reg': l2_reg
        }
        
        self.model = self._build_model()
    
    def _build_model(self) -> Sequential:
        """
        Construye la arquitectura del modelo LSTM.
        
        Returns:
            Modelo Keras compilado
        """
        model = Sequential()
        
        lstm_units = self.config['lstm_units']
        dropout_rate = self.config['dropout_rate']
        dense_units = self.config['dense_units']
        l2_reg = self.config['l2_reg']
        use_bidirectional = self.config['use_bidirectional']
        use_batch_norm = self.config['use_batch_norm']
        
        # Regularizador
        regularizer = l2(l2_reg) if l2_reg > 0 else None
        
        # Capas LSTM
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # True excepto para última LSTM
            
            lstm_layer = LSTM(
                units=units,
                return_sequences=return_sequences,
                kernel_regularizer=regularizer,
                input_shape=self.input_shape if i == 0 else None
            )
            
            if use_bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            
            if i == 0:
                model.add(lstm_layer)
            else:
                model.add(lstm_layer)
            
            if use_batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rate))
        
        # Capas Dense
        for units in dense_units:
            model.add(Dense(units, activation='relu', kernel_regularizer=regularizer))
            model.add(Dropout(dropout_rate / 2))
        
        # Capa de salida
        model.add(Dense(self.config['output_units']))
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def summary(self):
        """Muestra el resumen del modelo."""
        return self.model.summary()
    
    def get_model(self) -> Sequential:
        """Retorna el modelo Keras."""
        return self.model
    
    def save(self, filepath: str):
        """
        Guarda el modelo en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """
        Carga un modelo desde disco.
        
        Args:
            filepath: Ruta del modelo guardado
            
        Returns:
            Instancia de LSTMModel con el modelo cargado
        """
        from tensorflow.keras.models import load_model
        
        instance = cls.__new__(cls)
        instance.model = load_model(filepath)
        instance.input_shape = instance.model.input_shape[1:]
        instance.config = {}
        
        return instance


def create_lstm_model(
    input_shape: Tuple[int, int],
    complexity: str = 'medium'
) -> LSTMModel:
    """
    Crea un modelo LSTM con configuración predefinida.
    
    Args:
        input_shape: Tupla (time_steps, features)
        complexity: Nivel de complejidad ('simple', 'medium', 'complex')
        
    Returns:
        Instancia de LSTMModel configurada
    """
    configs = {
        'simple': {
            'lstm_units': [32],
            'dropout_rate': 0.2,
            'dense_units': [],
            'learning_rate': 0.001
        },
        'medium': {
            'lstm_units': [50, 50],
            'dropout_rate': 0.2,
            'dense_units': [25],
            'learning_rate': 0.001
        },
        'complex': {
            'lstm_units': [100, 50, 50],
            'dropout_rate': 0.3,
            'dense_units': [50, 25],
            'learning_rate': 0.0005,
            'use_batch_norm': True
        }
    }
    
    if complexity not in configs:
        raise ValueError(f"Complejidad '{complexity}' no válida. Use: simple, medium, complex")
    
    config = configs[complexity]
    
    return LSTMModel(input_shape=input_shape, **config)


def create_stacked_lstm(
    input_shape: Tuple[int, int],
    n_layers: int = 3,
    units_per_layer: int = 50,
    dropout_rate: float = 0.2
) -> LSTMModel:
    """
    Crea un modelo LSTM apilado con capas idénticas.
    
    Args:
        input_shape: Tupla (time_steps, features)
        n_layers: Número de capas LSTM
        units_per_layer: Unidades en cada capa
        dropout_rate: Tasa de dropout
        
    Returns:
        Instancia de LSTMModel
    """
    lstm_units = [units_per_layer] * n_layers
    
    return LSTMModel(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_units=[units_per_layer // 2]
    )


def create_bidirectional_lstm(
    input_shape: Tuple[int, int],
    lstm_units: List[int] = [64, 32],
    dropout_rate: float = 0.2
) -> LSTMModel:
    """
    Crea un modelo LSTM bidireccional.
    
    Args:
        input_shape: Tupla (time_steps, features)
        lstm_units: Lista de unidades por capa
        dropout_rate: Tasa de dropout
        
    Returns:
        Instancia de LSTMModel bidireccional
    """
    return LSTMModel(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_units=[32],
        use_bidirectional=True
    )


class MultiStepLSTM(LSTMModel):
    """
    Modelo LSTM para predicciones de múltiples pasos adelante.
    
    Útil cuando necesitas predecir varios valores futuros en una sola predicción.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        forecast_horizon: int = 7,
        **kwargs
    ):
        """
        Inicializa el modelo multi-step.
        
        Args:
            input_shape: Tupla (time_steps, features)
            forecast_horizon: Número de pasos a predecir
            **kwargs: Argumentos adicionales para LSTMModel
        """
        self.forecast_horizon = forecast_horizon
        super().__init__(input_shape=input_shape, output_units=forecast_horizon, **kwargs)


def get_model_config_summary(model: LSTMModel) -> str:
    """
    Genera un resumen legible de la configuración del modelo.
    
    Args:
        model: Instancia de LSTMModel
        
    Returns:
        String con el resumen de configuración
    """
    config = model.config
    
    summary = f"""
╔══════════════════════════════════════════════════════════╗
║              CONFIGURACIÓN DEL MODELO LSTM               ║
╠══════════════════════════════════════════════════════════╣
║ Input Shape:        {str(model.input_shape):<37} ║
║ Capas LSTM:         {str(config.get('lstm_units', 'N/A')):<37} ║
║ Dropout Rate:       {config.get('dropout_rate', 'N/A'):<37} ║
║ Capas Dense:        {str(config.get('dense_units', 'N/A')):<37} ║
║ Output Units:       {config.get('output_units', 'N/A'):<37} ║
║ Learning Rate:      {config.get('learning_rate', 'N/A'):<37} ║
║ Bidireccional:      {str(config.get('use_bidirectional', False)):<37} ║
║ Batch Norm:         {str(config.get('use_batch_norm', False)):<37} ║
║ L2 Regularization:  {config.get('l2_reg', 0.0):<37} ║
╚══════════════════════════════════════════════════════════╝
"""
    return summary



