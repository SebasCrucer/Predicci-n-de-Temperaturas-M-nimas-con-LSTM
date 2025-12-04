"""
Módulo de Análisis Exploratorio de Datos (EDA) para series temporales.

Este módulo proporciona funciones para:
- Visualización de series temporales
- Descomposición de series (tendencia, estacionalidad, residuos)
- Tests de estacionariedad (ADF)
- Gráficos de autocorrelación (ACF/PACF)
- Estadísticas descriptivas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Optional, Tuple, Dict
import warnings

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class TimeSeriesEDA:
    """
    Clase para realizar análisis exploratorio de series temporales.
    
    Attributes:
        data (pd.Series): Serie temporal a analizar
        name (str): Nombre de la serie para títulos de gráficos
    """
    
    def __init__(self, data: pd.Series, name: str = "Serie Temporal"):
        """
        Inicializa el analizador EDA.
        
        Args:
            data: Serie temporal (pd.Series con índice de fechas preferiblemente)
            name: Nombre descriptivo de la serie
        """
        self.data = data
        self.name = name
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Calcula estadísticas descriptivas de la serie.
        
        Returns:
            DataFrame con estadísticas descriptivas
        """
        stats = {
            'Conteo': len(self.data),
            'Media': self.data.mean(),
            'Mediana': self.data.median(),
            'Desv. Estándar': self.data.std(),
            'Varianza': self.data.var(),
            'Mínimo': self.data.min(),
            'Máximo': self.data.max(),
            'Rango': self.data.max() - self.data.min(),
            'Asimetría': self.data.skew(),
            'Curtosis': self.data.kurtosis(),
            'Q1 (25%)': self.data.quantile(0.25),
            'Q3 (75%)': self.data.quantile(0.75),
            'IQR': self.data.quantile(0.75) - self.data.quantile(0.25)
        }
        
        df_stats = pd.DataFrame(list(stats.items()), columns=['Estadística', 'Valor'])
        return df_stats
    
    def plot_series(
        self,
        figsize: Tuple[int, int] = (14, 5),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza la serie temporal completa.
        
        Args:
            figsize: Tamaño de la figura
            title: Título del gráfico
            save_path: Ruta para guardar la imagen
            
        Returns:
            Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.data.index, self.data.values, linewidth=1, color='#2E86AB')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_title(title or f'{self.name} - Serie Temporal', fontsize=14, fontweight='bold')
        
        # Añadir media móvil
        if len(self.data) > 30:
            ma = self.data.rolling(window=30).mean()
            ax.plot(self.data.index, ma, color='#E94F37', linewidth=2, 
                   label='Media Móvil (30)', alpha=0.8)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        return fig
    
    def plot_distribution(
        self,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza la distribución de los valores.
        
        Args:
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la imagen
            
        Returns:
            Figura de matplotlib
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histograma
        axes[0].hist(self.data, bins=50, edgecolor='white', color='#2E86AB', alpha=0.7)
        axes[0].axvline(self.data.mean(), color='#E94F37', linestyle='--', 
                       linewidth=2, label=f'Media: {self.data.mean():.2f}')
        axes[0].set_xlabel('Valor')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Valores')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.data, vert=True)
        axes[1].set_ylabel('Valor')
        axes[1].set_title('Box Plot')
        
        plt.suptitle(f'{self.name} - Análisis de Distribución', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def decompose(
        self,
        period: Optional[int] = None,
        model: str = 'additive',
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Descompone la serie en tendencia, estacionalidad y residuos.
        
        Args:
            period: Período de la estacionalidad (si None, intenta detectarlo)
            model: Tipo de descomposición ('additive' o 'multiplicative')
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la imagen
            
        Returns:
            Figura de matplotlib
        """
        if period is None:
            # Intentar detectar período basado en índice de fechas
            if isinstance(self.data.index, pd.DatetimeIndex):
                freq = pd.infer_freq(self.data.index)
                if freq:
                    if 'D' in freq:
                        period = 7  # Semanal
                    elif 'M' in freq:
                        period = 12  # Anual
                    elif 'H' in freq:
                        period = 24  # Diario
            if period is None:
                period = min(len(self.data) // 4, 30)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomposition = seasonal_decompose(self.data, model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Serie original
        axes[0].plot(self.data.index, self.data.values, color='#2E86AB')
        axes[0].set_ylabel('Original')
        axes[0].set_title(f'{self.name} - Descomposición ({model})', fontsize=14, fontweight='bold')
        
        # Tendencia
        axes[1].plot(self.data.index, decomposition.trend, color='#E94F37')
        axes[1].set_ylabel('Tendencia')
        
        # Estacionalidad
        axes[2].plot(self.data.index, decomposition.seasonal, color='#28A745')
        axes[2].set_ylabel('Estacionalidad')
        
        # Residuos
        axes[3].plot(self.data.index, decomposition.resid, color='#6C757D')
        axes[3].set_ylabel('Residuos')
        axes[3].set_xlabel('Fecha')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def adf_test(self) -> Dict:
        """
        Realiza el test de Dickey-Fuller Aumentado (ADF) para estacionariedad.
        
        Returns:
            Diccionario con resultados del test
        """
        result = adfuller(self.data.dropna(), autolag='AIC')
        
        output = {
            'Estadístico ADF': result[0],
            'p-valor': result[1],
            'Lags utilizados': result[2],
            'Observaciones': result[3],
            'Valores críticos': result[4],
            'Es estacionaria': result[1] < 0.05
        }
        
        print("=" * 50)
        print("TEST DE DICKEY-FULLER AUMENTADO (ADF)")
        print("=" * 50)
        print(f"Estadístico ADF: {result[0]:.6f}")
        print(f"p-valor: {result[1]:.6f}")
        print(f"Lags utilizados: {result[2]}")
        print(f"Observaciones: {result[3]}")
        print("\nValores Críticos:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        print("-" * 50)
        
        if result[1] < 0.05:
            print("✓ La serie ES ESTACIONARIA (p-valor < 0.05)")
        else:
            print("✗ La serie NO es estacionaria (p-valor >= 0.05)")
        print("=" * 50)
        
        return output
    
    def plot_acf_pacf(
        self,
        lags: int = 40,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Grafica las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
        
        Args:
            lags: Número de lags a mostrar
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la imagen
            
        Returns:
            Figura de matplotlib
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ACF
        plot_acf(self.data.dropna(), lags=lags, ax=axes[0], color='#2E86AB')
        axes[0].set_title('Función de Autocorrelación (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('Autocorrelación')
        
        # PACF
        plot_pacf(self.data.dropna(), lags=lags, ax=axes[1], color='#E94F37', method='ywm')
        axes[1].set_title('Función de Autocorrelación Parcial (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('Autocorrelación Parcial')
        
        plt.suptitle(f'{self.name} - Análisis de Autocorrelación', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_statistics(
        self,
        window: int = 30,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza estadísticas móviles (media y desviación estándar).
        
        Args:
            window: Tamaño de la ventana para cálculos móviles
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la imagen
            
        Returns:
            Figura de matplotlib
        """
        rolling_mean = self.data.rolling(window=window).mean()
        rolling_std = self.data.rolling(window=window).std()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Serie original con media móvil
        axes[0].plot(self.data.index, self.data.values, label='Original', 
                    color='#2E86AB', alpha=0.6)
        axes[0].plot(self.data.index, rolling_mean, label=f'Media Móvil ({window})', 
                    color='#E94F37', linewidth=2)
        axes[0].set_ylabel('Valor')
        axes[0].set_title(f'{self.name} - Media Móvil', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # Desviación estándar móvil
        axes[1].plot(self.data.index, rolling_std, color='#28A745', linewidth=2)
        axes[1].set_ylabel('Desviación Estándar')
        axes[1].set_xlabel('Fecha')
        axes[1].set_title(f'Desviación Estándar Móvil ({window})', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def full_eda_report(
        self,
        output_dir: str = 'outputs',
        period: Optional[int] = None
    ) -> Dict:
        """
        Genera un reporte EDA completo con todas las visualizaciones.
        
        Args:
            output_dir: Directorio donde guardar las gráficas
            period: Período para descomposición estacional
            
        Returns:
            Diccionario con resultados del análisis
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print(f"REPORTE EDA: {self.name}")
        print("=" * 60)
        
        # Estadísticas descriptivas
        print("\n📊 ESTADÍSTICAS DESCRIPTIVAS")
        stats = self.summary_statistics()
        print(stats.to_string(index=False))
        
        # Test ADF
        print("\n📈 TEST DE ESTACIONARIEDAD")
        adf_results = self.adf_test()
        
        # Generar gráficas
        print("\n📉 GENERANDO VISUALIZACIONES...")
        
        self.plot_series(save_path=f'{output_dir}/01_serie_temporal.png')
        print("  ✓ Serie temporal guardada")
        
        self.plot_distribution(save_path=f'{output_dir}/02_distribucion.png')
        print("  ✓ Distribución guardada")
        
        self.decompose(period=period, save_path=f'{output_dir}/03_descomposicion.png')
        print("  ✓ Descomposición guardada")
        
        self.plot_acf_pacf(save_path=f'{output_dir}/04_acf_pacf.png')
        print("  ✓ ACF/PACF guardados")
        
        self.plot_rolling_statistics(save_path=f'{output_dir}/05_rolling_stats.png')
        print("  ✓ Estadísticas móviles guardadas")
        
        print("\n" + "=" * 60)
        print(f"Reporte completado. Gráficas guardadas en: {output_dir}/")
        print("=" * 60)
        
        plt.close('all')
        
        return {
            'statistics': stats,
            'adf_test': adf_results
        }


def quick_eda(
    data: pd.Series,
    name: str = "Serie Temporal",
    output_dir: str = 'outputs'
) -> TimeSeriesEDA:
    """
    Función de conveniencia para ejecutar un EDA rápido.
    
    Args:
        data: Serie temporal
        name: Nombre de la serie
        output_dir: Directorio de salida
        
    Returns:
        Objeto TimeSeriesEDA con los resultados
    """
    eda = TimeSeriesEDA(data, name)
    eda.full_eda_report(output_dir)
    return eda



