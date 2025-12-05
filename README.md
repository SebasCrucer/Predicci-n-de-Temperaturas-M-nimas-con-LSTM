# Predicción de Temperaturas Mínimas con LSTM

Modelo de series temporales para predecir temperaturas mínimas diarias usando redes neuronales LSTM.

## Despliegue 
link aquí 

## Dataset

**Temperaturas Mínimas Diarias - Melbourne, Australia (1981-1990)**
- 3,650 observaciones
- Período: 10 años completos
- Variable: Temperatura mínima diaria (°C)

## Resultados del Modelo Optimizado

| Métrica | Valor |
|---------|-------|
| **RMSE** | 2.23 °C |
| **MAE** | 1.75 °C |
| **R²** | 0.71 |
| **MAPE** | 21.26% |

### Configuración Óptima (encontrada mediante grid search)
- Sequence Length: 60 días
- LSTM Units: [64]
- Dropout: 0.3
- Learning Rate: 0.001

## Estructura del Proyecto

```
proyecto_final/
├── data/
│   └── 1_Daily_minimum_temps.xls   # Dataset original (3,650 obs)
├── notebooks/
│   └── analysis.ipynb              # Notebook de análisis completo
├── src/
│   ├── data_loader.py              # Carga y preprocesamiento
│   ├── eda.py                      # Análisis exploratorio
│   ├── model.py                    # Arquitectura LSTM
│   ├── train.py                    # Pipeline de entrenamiento
│   ├── evaluate.py                 # Evaluación y métricas
│   └── optimizer.py                # Grid search
├── outputs/
│   ├── best_model_final.keras      # Modelo optimizado (19K params)
│   ├── final_metrics.csv           # Métricas finales
│   ├── optimization_results.csv    # Resultados grid search
│   ├── predictions_optimized.png   # Gráfico predicciones
│   ├── scatter_optimized.png       # Scatter plot
│   ├── error_analysis_optimized.png # Análisis de errores
│   └── training_history_optimized.png # Historia entrenamiento
├── app_streamlit.py                # Aplicación web interactiva
├── requirements.txt                # Dependencias Python
├── README.md                       # Este documento
└── reporte/
    └── reporte.md         #Reporte técnico
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Jupyter Notebook (Recomendado)
```bash
jupyter notebook notebooks/analysis.ipynb
```

### Script de Python
```python
import pandas as pd
from src.data_loader import TimeSeriesDataLoader
from src.model import LSTMModel
from src.train import train_model
from src.evaluate import evaluate_model

# Cargar datos
df = pd.read_csv('data/1_Daily_minimum_temps.xls')
df['Temp'] = df['Temp'].astype(str).str.replace('?', '-', regex=False)
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df.set_index('Date', inplace=True)
df = df.dropna()

# Preparar datos
df_reset = df.reset_index()
df_reset.columns = ['fecha', 'valor']
loader = TimeSeriesDataLoader(sequence_length=60)
loader.load_from_dataframe(df_reset, target_column='valor')
X_train, X_test, y_train, y_test = loader.prepare_data(test_size=0.2)

# Crear modelo optimizado
model = LSTMModel(
    input_shape=(60, 1),
    lstm_units=[64],
    dropout_rate=0.3,
    dense_units=[32],
    learning_rate=0.001
)

# Entrenar
train_model(model, X_train, y_train, epochs=150)

# Evaluar
results = evaluate_model(model, X_test, y_test, loader.scaler)
```

## Dependencias

- tensorflow >= 2.0
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- statsmodels

## Autores

Arturo Cantú Olivarez | Diego Sebastian Cruz Cervantes
Proyecto Final - Modelos Predictivos


