# 📋 Guía de Uso - Aplicación Streamlit

## 🚀 Inicio Rápido

### Ejecutar la Aplicación

```bash
# Navegar al directorio del proyecto
cd "c:\Users\cantu\OneDrive\Escritorio\Universidad\Semestre 5\Modelos Predictivos\proyectoFinal"

# Ejecutar con Python
python -m streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en tu navegador en: **http://localhost:8501**

---

## 📱 Secciones de la Aplicación

### 1. 📊 Métricas y Resultados

**Qué verás:**
- Métricas principales del modelo (RMSE, MAE, R², MAPE)
- Gráfico temporal: Predicciones vs Valores Reales
- Gráfico de dispersión con R²
- Distribución de errores

**Interacción:**
- Hover sobre los gráficos para ver valores exactos
- Zoom con scroll o selección de área
- Pan arrastrando el gráfico

---

### 2. 🎯 Predicción Interactiva

#### Opción 1: Predicción Rápida 🚀

**Pasos:**
1. Selecciona una fecha del calendario
2. Haz clic en "🔮 Predecir"
3. Ve la predicción, el valor real, y el error
4. Examina el gráfico de la secuencia de 60 días

**Ejemplo:**
```
Selecciona: 1985-06-15
Resultado: Temperatura predicha para el día siguiente
```

#### Opción 2: Predicción Personalizada ✏️

**Pasos:**
1. Ingresa 60 valores de temperatura separados por comas
2. Haz clic en "🔮 Predecir"
3. Ve la predicción destacada
4. Examina estadísticas y gráfico de tu secuencia

**Ejemplo de entrada:**
```
10.5, 11.2, 9.8, 12.1, 13.4, 14.2, 13.1, 15.3, 14.8, 12.5,
11.9, 13.7, 14.5, 12.8, 11.4, 13.2, 15.1, 14.3, 13.6, 12.9,
11.7, 13.5, 14.9, 13.8, 12.3, 14.1, 15.6, 14.7, 13.4, 12.8,
11.5, 13.9, 15.3, 14.2, 12.6, 14.4, 16.1, 15.2, 13.9, 13.1,
12.2, 14.8, 16.5, 15.5, 13.7, 15.2, 17.3, 16.4, 14.8, 13.5,
12.9, 15.6, 17.8, 16.8, 14.5, 16.1, 18.2, 17.5, 15.9, 14.2
```

---

### 3. 📈 Visualizaciones

**Contenido:**
- Imágenes de todas las visualizaciones generadas durante el entrenamiento
- Tabla de resultados del grid search de hiperparámetros
- Gráfico interactivo de RMSE por configuración

**Navegación:**
- Usa las pestañas para ver diferentes visualizaciones
- Scroll para ver la tabla completa de resultados

---

### 4. ⚙️ Configuración del Modelo

**Información disponible:**
- Hiperparámetros óptimos utilizados
- Estadísticas del dataset
- Arquitectura detallada del modelo LSTM
- Proceso de optimización

---

## 💡 Consejos de Uso

### Para Predicción Personalizada

✅ **Formato correcto:**
```
10.5, 11.2, 9.8, 12.1, ...
```

❌ **Formato incorrecto:**
```
10.5; 11.2; 9.8  (usa comas, no puntos y coma)
10.5 11.2 9.8    (necesita comas)
```

### Validaciones

- ✅ Debes ingresar **exactamente 60 valores**
- ✅ Todos deben ser **números válidos**
- ⚠️ Valores fuera de -50 a 50°C mostrarán un warning
- ❌ Menos o más de 60 valores mostrará un error

---

## 🎨 Características Interactivas

### Gráficos Plotly

Todos los gráficos son interactivos:

- **Zoom**: Selecciona un área arrastrando, o usa scroll
- **Pan**: Arrastra el gráfico para moverte
- **Reset**: Doble clic para volver a la vista original
- **Hover**: Pasa el mouse para ver valores exactos
- **Download**: Botón de cámara para guardar como imagen

### Navegación

- **Sidebar**: Panel izquierdo para cambiar de sección
- **Tabs**: Pestañas dentro de cada sección
- **Botones**: Formularios para hacer predicciones

---

## 🔧 Solución de Problemas

### La aplicación no inicia

```bash
# Verifica que las dependencias estén instaladas
pip install -r requirements.txt

# Ejecuta con python -m
python -m streamlit run app_streamlit.py
```

### Error al cargar el modelo

**Causa:** No se encuentra el archivo `best_model_final.keras`

**Solución:** Asegúrate de estar en el directorio correcto y que exista la carpeta `outputs/` con los archivos necesarios

### Errores de predicción

**Causa:** Formato incorrecto de los valores

**Solución:** 
- Usa exactamente 60 valores
- Separa con comas
- No uses espacios innecesarios
- Verifica que todos sean números

---

## 📊 Especificaciones Técnicas

### Requisitos del Modelo

- **Input**: Secuencia de 60 temperaturas consecutivas (°C)
- **Output**: Predicción de temperatura del día 61 (°C)
- **Rango típico**: -10°C a 25°C (dataset de Melbourne)

### Performance

- **Tiempo de carga inicial**: ~2-3 segundos
- **Tiempo de predicción**: <100ms (después de cargar)
- **Uso de memoria**: ~500MB (modelo + datos cacheados)

---

## ✨ Funcionalidades Destacadas

1. ⚡ **Predicciones en tiempo real** con caché inteligente
2. 📊 **Gráficos interactivos** con Plotly
3. ✅ **Validación robusta** de inputs
4. 🎨 **Diseño moderno** y responsive
5. 📱 **Navegación intuitiva** con sidebar
6. 💾 **Persistencia** de carga de modelo (una sola vez)

---

## 🎯 Casos de Uso

### 1. Exploración de Resultados
- Ve las métricas del modelo
- Compara predicciones vs valores reales
- Analiza la distribución de errores

### 2. Predicción Rápida
- Selecciona una fecha histórica
- Ve cómo el modelo predice vs realidad
- Entiende el comportamiento del modelo

### 3. Experimentación
- Ingresa tus propias secuencias
- Prueba tendencias ascendentes/descendentes
- Experimenta con valores extremos

### 4. Presentación
- Muestra el trabajo del proyecto
- Demuestra capacidades del modelo
- Explica el proceso de optimización

---

## 🎓 Para más información

- **Dataset**: Melbourne Daily Minimum Temperatures (1981-1990)
- **Modelo**: LSTM con 64 unidades, dropout 0.3
- **Optimización**: Grid Search con 36 configuraciones
- **Métricas**: RMSE: 2.23°C, MAE: 1.75°C, R²: 0.71

---

**¡Disfruta explorando las predicciones! 🌡️🚀**
