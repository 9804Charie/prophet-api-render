import pandas as pd
from prophet import Prophet
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Para validación de datos
from typing import List, Dict, Any
import uvicorn
import logging # Para ver mensajes
import os
from dotenv import load_dotenv

# Cargar variables de entorno (opcional, para seguridad de claves si las usaras)
load_dotenv()

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Definir modelos de datos para la API (usando Pydantic) ---

class TimePoint(BaseModel):
    # Asegúrate que 'fecha' viene en un formato que pandas pueda entender (ej. 'YYYY-MM-DD')
    fecha: str
    # Asegúrate que 'cantidad' es numérica
    cantidad: float | int

class PredictionRequest(BaseModel):
    keyword: str # Recibimos la keyword para logging/contexto
    historical_data: List[TimePoint] # Una lista de puntos históricos {fecha, cantidad}
    prediction_periods: int = Field(gt=0) # Número de periodos a predecir (debe ser > 0)

class ForecastPoint(BaseModel):
    # Cambia los nombres si quieres que la respuesta JSON tenga otros nombres
    predicted_date: str
    predicted_value: float
    prediction_lower_bound: float # Opcional: Límite inferior del intervalo
    prediction_upper_bound: float # Opcional: Límite superior del intervalo

class PredictionResponse(BaseModel):
    status: str
    keyword: str
    forecast: List[ForecastPoint]

# --- Inicializar la aplicación FastAPI ---
app = FastAPI(
    title="API de Predicción con Prophet",
    description="Recibe datos históricos y devuelve predicciones usando Prophet.",
    version="1.0.0"
)

# --- Definir el Endpoint de Predicción ---

@app.post("/predict", response_model=PredictionResponse)
async def predict_keyword_trend(request: PredictionRequest):
    """
    Recibe datos históricos de una keyword y devuelve predicciones futuras.
    """
    logger.info(f"Recibida solicitud de predicción para keyword: {request.keyword}")

    # 1. Validar datos de entrada (FastAPI/Pydantic ya lo hacen parcialmente)
    if len(request.historical_data) < 2: # Prophet necesita al menos 2 puntos
        logger.error("Datos históricos insuficientes (se necesitan al menos 2 puntos).")
        raise HTTPException(status_code=400, detail="Datos históricos insuficientes (se necesitan al menos 2 puntos).")

    # 2. Preparar DataFrame para Prophet
    try:
        # Convertir la lista de dicts a DataFrame
        history_list = [item.dict() for item in request.historical_data]
        df = pd.DataFrame(history_list)

        # Renombrar columnas a 'ds' (datestamp) y 'y' (valor) como requiere Prophet
        df = df.rename(columns={'fecha': 'ds', 'cantidad': 'y'})

        # Convertir 'ds' a formato datetime (crucial!)
        # Intentamos inferir el formato, ajusta 'format' si tus fechas son específicas
        df['ds'] = pd.to_datetime(df['ds'])

        # Eliminar filas con fechas o valores nulos después de la conversión
        df = df.dropna(subset=['ds', 'y'])

        if len(df) < 2:
             logger.error("Datos históricos insuficientes después de la limpieza.")
             raise HTTPException(status_code=400, detail="Datos históricos insuficientes después de la limpieza.")

        # Ordenar por fecha (importante para Prophet)
        df = df.sort_values(by='ds')

        logger.info(f"DataFrame preparado con {len(df)} puntos para keyword: {request.keyword}")

    except Exception as e:
        logger.exception(f"Error preparando el DataFrame para {request.keyword}: {e}")
        raise HTTPException(status_code=400, detail=f"Error en el formato de datos históricos: {e}")

    # 3. Entrenar Modelo Prophet y Predecir
    try:
        # Instanciar Prophet (puedes ajustar parámetros aquí si es necesario)
        # ej. Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model = Prophet()

        # Entrenar el modelo
        model.fit(df)

        # Crear dataframe futuro para las predicciones
        # 'freq' podría ser 'D' (diario), 'W' (semanal, domingo), 'W-MON' (semanal, lunes), 'MS' (inicio de mes)
        # ¡AJUSTA 'freq' según la frecuencia de tus datos históricos!
        future = model.make_future_dataframe(periods=request.prediction_periods, freq='W-SUN') # Ejemplo: Semanal terminando en Domingo

        # Realizar la predicción
        forecast = model.predict(future)

        logger.info(f"Predicción generada para {request.keyword}")

        # Seleccionar solo las predicciones futuras (excluir el historial)
        # forecast['ds'] ya es datetime, lo convertimos a string para JSON
        future_forecast = forecast[forecast['ds'] > df['ds'].max()].copy()
        future_forecast['ds_str'] = future_forecast['ds'].dt.strftime('%Y-%m-%d')

        # 4. Formatear la Respuesta
        response_forecast: List[ForecastPoint] = []
        for _, row in future_forecast.iterrows():
            response_forecast.append(ForecastPoint(
                predicted_date=row['ds_str'],
                predicted_value=row['yhat'],
                prediction_lower_bound=row['yhat_lower'],
                prediction_upper_bound=row['yhat_upper']
            ))

        return PredictionResponse(
            status="success",
            keyword=request.keyword,
            forecast=response_forecast
        )

    except Exception as e:
        logger.exception(f"Error durante el entrenamiento o predicción de Prophet para {request.keyword}: {e}")
        raise HTTPException(status_code=500, detail=f"Error del modelo Prophet: {e}")

# --- Ejecutar el servidor API (si ejecutas este script directamente) ---
if __name__ == "__main__":
    # Lee el puerto del entorno o usa 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Iniciando servidor Uvicorn en http://127.0.0.1:{port}")
    # 'reload=True' es útil para desarrollo, quítalo para producción
    uvicorn.run("prophet_api:app", host="0.0.0.0", port=port, reload=True)