"""
Módulo de carga y preprocesamiento de datos para series temporales.

Este módulo proporciona funciones para:
- Cargar datos desde CSV/Excel
- Preprocesar y normalizar datos
- Crear secuencias para entrenamiento de LSTM
- Dividir datos en conjuntos de entrenamiento y prueba
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Union


class TimeSeriesDataLoader:
    """
    Clase para cargar y preprocesar datos de series temporales para LSTM.
    
    Attributes:
        data (pd.DataFrame): DataFrame con los datos cargados
        scaler (MinMaxScaler): Escalador para normalización
        sequence_length (int): Longitud de las secuencias para LSTM
    """
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        sequence_length: int = 60,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None
    ):
        """
        Inicializa el cargador de datos.
        
        Args:
            file_path: Ruta al archivo de datos (CSV o Excel)
            sequence_length: Número de pasos de tiempo para cada secuencia
            target_column: Nombre de la columna objetivo (si None, usa la primera columna numérica)
            date_column: Nombre de la columna de fechas (si None, intenta detectarla)
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.date_column = date_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV o Excel.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame con los datos cargados
        """
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError("Formato de archivo no soportado. Use CSV o Excel.")
        
        # Detectar y configurar columna de fecha
        self._setup_date_index()
        
        # Detectar columna objetivo si no se especificó
        if self.target_column is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.target_column = numeric_cols[0]
            else:
                raise ValueError("No se encontraron columnas numéricas en los datos.")
        
        print(f"Datos cargados: {len(self.data)} registros")
        print(f"Columna objetivo: {self.target_column}")
        
        return self.data
    
    def _setup_date_index(self):
        """Configura la columna de fecha como índice del DataFrame."""
        if self.date_column:
            date_col = self.date_column
        else:
            # Intentar detectar columna de fecha
            for col in self.data.columns:
                if 'date' in col.lower() or 'fecha' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break
            else:
                # Si no encuentra, usar la primera columna si parece ser fecha
                first_col = self.data.columns[0]
                try:
                    pd.to_datetime(self.data[first_col].iloc[:5])
                    date_col = first_col
                except:
                    date_col = None
        
        if date_col:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data.set_index(date_col, inplace=True)
            self.data.sort_index(inplace=True)
            self.date_column = date_col
    
    def load_from_dataframe(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Carga datos desde un DataFrame existente.
        
        Args:
            df: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            
        Returns:
            DataFrame procesado
        """
        self.data = df.copy()
        self.target_column = target_column
        return self.data
    
    def normalize_data(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normaliza los datos usando MinMaxScaler.
        
        Args:
            data: Datos a normalizar (si None, usa self.data[target_column])
            
        Returns:
            Datos normalizados
        """
        if data is None:
            data = self.data[self.target_column].values.reshape(-1, 1)
        
        self.scaled_data = self.scaler.fit_transform(data)
        return self.scaled_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Revierte la normalización de los datos.
        
        Args:
            data: Datos normalizados
            
        Returns:
            Datos en escala original
        """
        return self.scaler.inverse_transform(data.reshape(-1, 1))
    
    def create_sequences(
        self,
        data: Optional[np.ndarray] = None,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias de datos para entrenamiento de LSTM.
        
        Args:
            data: Datos normalizados (si None, usa self.scaled_data)
            sequence_length: Longitud de cada secuencia (si None, usa self.sequence_length)
            
        Returns:
            Tupla (X, y) donde X son las secuencias de entrada e y son los valores objetivo
        """
        if data is None:
            if self.scaled_data is None:
                self.normalize_data()
            data = self.scaled_data
        
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i, 0])
            y.append(data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape para LSTM: [samples, time_steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Nota: Para series temporales, NO se mezclan los datos aleatoriamente,
        se respeta el orden temporal.
        
        Args:
            X: Secuencias de entrada
            y: Valores objetivo
            test_size: Proporción de datos para prueba (default 0.2)
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Datos de entrenamiento: {len(X_train)} secuencias")
        print(f"Datos de prueba: {len(X_test)} secuencias")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(
        self,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline completo de preparación de datos.
        
        Args:
            test_size: Proporción de datos para prueba
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Use load_data() primero.")
        
        # Normalizar
        self.normalize_data()
        
        # Crear secuencias
        X, y = self.create_sequences()
        
        # Dividir en train/test
        return self.train_test_split(X, y, test_size)
    
    def get_dates_for_predictions(self, test_size: float = 0.2) -> pd.DatetimeIndex:
        """
        Obtiene las fechas correspondientes a las predicciones del conjunto de prueba.
        
        Args:
            test_size: Proporción de datos usados para prueba
            
        Returns:
            Índice de fechas para el conjunto de prueba
        """
        if not isinstance(self.data.index, pd.DatetimeIndex):
            return None
        
        total_sequences = len(self.data) - self.sequence_length
        split_idx = int(total_sequences * (1 - test_size))
        
        # Las fechas comienzan después de sequence_length + split_idx
        start_idx = self.sequence_length + split_idx
        
        return self.data.index[start_idx:]



