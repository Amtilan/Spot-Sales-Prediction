from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional, Dict, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel
from threading import Lock
import pandas as pd
import numpy as np
import uvicorn
import logging
import shutil
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Restaurant Order Prediction API")

# Initialize scheduler
scheduler = BackgroundScheduler()

# CORS settings
origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    "http://localhost:3001",
    "https://localhost:3001",
    "spot-sales-prediction-reworked.vercel.app",
    "https://spot-sales-prediction-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resources
data_lock = Lock()
df = None
models = None
cache: Dict[str, Dict] = {}
CACHE_EXPIRATION_HOURS = 24  # Cache expiration
DATA_FILE = "the_burger_spot.csv"  # Data file path

# Pydantic models
class PredictionRequest(BaseModel):
    area: str
    range: Literal["1d", "1w", "2w", "1m"]

class PredictionItem(BaseModel):
    Category: str
    Item: str
    Type_of_Order: str
    Predicted_Orders: int
    Date: str

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    total_orders: int
    area: str
    date_range: Dict[str, str]
    daily_totals: Dict[str, int]

class MenuResponse(BaseModel):
    categories: Dict[str, List[str]]
    total_items: int

class UploadResponse(BaseModel):
    message: str
    rows_processed: int

# Utility functions
def clear_cache():
    global cache
    cache = {}
    logger.info("Cache cleared")

def load_data(file_path: str = DATA_FILE):
    global df, models
    try:
        logger.info(f"Loading data from {file_path}")
        new_df = pd.read_csv(file_path)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        df = new_df
        models = train_area_models(new_df)
        logger.info(f"Data loaded and models trained successfully: {len(df)} rows")
        return len(df)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_area_models(data: pd.DataFrame) -> Dict:
    models = {}
    data = data.copy()
    
    def create_features(data):
        features = pd.DataFrame()
        features['Year'] = data['Date'].dt.year
        features['Month'] = data['Date'].dt.month
        features['DayOfWeek'] = data['Date'].dt.dayofweek
        features['DayOfMonth'] = data['Date'].dt.day
        return features

    for area in data['Area'].unique():
        area_data = data[data['Area'] == area]
        for item in area_data['Item'].unique():
            for order_type in ['Takeaway', 'Dining']:
                filtered_data = area_data[
                    (area_data['Item'] == item) & 
                    (area_data['Type of Order'] == order_type)
                ]
                if not filtered_data.empty:
                    X = create_features(filtered_data)
                    y = filtered_data['Orders']
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    models[(area, item, order_type)] = model
    return models

def get_date_range(start_date: datetime, range_type: str) -> List[datetime]:
    range_mappings = {
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "2w": timedelta(weeks=2),
        "1m": timedelta(days=30)
    }
    delta = range_mappings[range_type]
    end_date = start_date + delta - timedelta(days=1)  # Subtract 1 day to ensure correct range
    return pd.date_range(start=start_date, end=end_date, freq='D')

def predict_for_area_and_range(area: str, target_date: datetime, range_type: str) -> Optional[pd.DataFrame]:
    if df is None or models is None:
        raise HTTPException(status_code=500, detail="Data or models not loaded")

    date_range = get_date_range(target_date, range_type)
    area_data = df[df['Area'] == area]
    if area_data.empty:
        return None

    predictions = []
    for date in date_range:
        features = pd.DataFrame({
            'Year': [date.year],
            'Month': [date.month],
            'DayOfWeek': [date.dayofweek],
            'DayOfMonth': [date.day]
        })
        for item in area_data['Item'].unique():
            for order_type in ['Takeaway', 'Dining']:
                model_key = (area, item, order_type)
                if model_key in models:
                    prediction = models[model_key].predict(features)[0]
                    if prediction > 0:
                        predictions.append({
                            "Category": area_data[area_data['Item'] == item]['Category'].iloc[0],
                            "Item": item,
                            "Type_of_Order": order_type,
                            "Predicted_Orders": round(prediction),
                            "Date": date.strftime("%Y-%m-%d")
                        })
    return pd.DataFrame(predictions)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if df is None:
        raise HTTPException(status_code=404, detail="Data not loaded")

    target_date = datetime.now()  # Use the current date
    cache_key = f"{request.area}_{target_date.strftime('%Y-%m-%d')}_{request.range}"
    if cache_key in cache and cache[cache_key]['expiration'] > datetime.now():
        logger.info(f"Cache hit for {cache_key}")
        return cache[cache_key]['data']

    logger.info(f"Cache miss for {cache_key}")
    predictions_df = predict_for_area_and_range(request.area, target_date, request.range)
    if predictions_df is None or predictions_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for area: {request.area}")

    daily_totals = predictions_df.groupby('Date')['Predicted_Orders'].sum().to_dict()
    response = PredictionResponse(
        predictions=predictions_df.to_dict("records"),
        total_orders=int(predictions_df['Predicted_Orders'].sum()),
        area=request.area,
        date_range={
            "start_date": target_date.strftime("%Y-%m-%d"),
            "end_date": (target_date + timedelta(days=len(daily_totals)-1)).strftime("%Y-%m-%d"),
        },
        daily_totals=daily_totals,
    )
    cache[cache_key] = {"data": response, "expiration": datetime.now() + timedelta(hours=CACHE_EXPIRATION_HOURS)}
    return response

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        with data_lock:
            # Save uploaded file
            with open(DATA_FILE, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Clear cache and reload data
            clear_cache()
            rows_processed = load_data()
            
            return UploadResponse(
                message="File uploaded and processed successfully",
                rows_processed=rows_processed
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.get("/areas")
async def get_areas():
    if df is None:
        raise HTTPException(status_code=404, detail="Data not loaded")
    return {"areas": sorted(df["Area"].unique().tolist())}

@app.on_event("startup")
async def on_startup():
    with data_lock:
        if os.path.exists(DATA_FILE):
            load_data()
        else:
            logger.warning("No initial data file found")
    scheduler.add_job(load_data, CronTrigger(hour=0), id="daily_reload")
    scheduler.start()

@app.on_event("shutdown")
async def on_shutdown():
    scheduler.shutdown()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))