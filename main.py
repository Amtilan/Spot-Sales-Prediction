from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, validator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Restaurant Order Prediction API")

# Initialize scheduler
scheduler = BackgroundScheduler()

# Global DataFrame
df = None

# CORS
origins = [
    "https://localhost:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache dictionary to store predictions
cache: Dict[str, Dict] = {}
CACHE_EXPIRATION_HOURS = 24  # Expire cached items after 1 day

# Pydantic models
class PredictionRequest(BaseModel):
    area: str
    target_date: str
    range: Literal["1d", "1w", "2w", "1m"]

    @validator("target_date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")

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

# Data loading function
def load_data():
    global df
    try:
        logger.info("Starting data reload...")
        new_df = pd.read_csv('the_burger_spot.csv')
        new_df = load_and_prepare_data(new_df)
        df = new_df
        logger.info("Data reload completed successfully")
    except Exception as e:
        logger.error(f"Error reloading data: {str(e)}")
        raise

def load_and_prepare_data(data):
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_menu_items(df):
    return df.groupby('Category')['Item'].unique().to_dict()

def train_area_models(df):
    models = {}
    
    def create_features(data):
        features = pd.DataFrame()
        features['Year'] = data['Date'].dt.year
        features['Month'] = data['Date'].dt.month
        features['DayOfWeek'] = data['Date'].dt.dayofweek
        features['DayOfMonth'] = data['Date'].dt.day
        return features
    
    for item in df['Item'].unique():
        for order_type in ['Takeaway', 'Dining']:
            item_data = df[(df['Item'] == item) & (df['Type of Order'] == order_type)]
            if len(item_data) > 0:
                X = create_features(item_data)
                y = item_data['Orders']
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                models[(item, order_type)] = model
    
    return models

def get_date_range(target_date: str, range_type: str) -> List[datetime]:
    start_date = datetime.strptime(target_date, "%Y-%m-%d")
    
    range_mappings = {
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "2w": timedelta(weeks=2),
        "1m": timedelta(days=30)
    }
    
    delta = range_mappings[range_type]
    end_date = start_date + delta
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return date_range

def predict_for_area_and_range(df: pd.DataFrame, area: str, target_date: str, range_type: str) -> Optional[pd.DataFrame]:
    area_data = df[df['Area'] == area].copy()
    
    if len(area_data) == 0:
        return None
    
    menu_items = get_menu_items(df)
    models = train_area_models(area_data)
    date_range = get_date_range(target_date, range_type)
    
    all_predictions = []
    
    for pred_date in date_range:
        pred_features = pd.DataFrame({
            'Year': [pred_date.year],
            'Month': [pred_date.month],
            'DayOfWeek': [pred_date.dayofweek],
            'DayOfMonth': [pred_date.day]
        })
        
        for category, items in menu_items.items():
            for item in items:
                for order_type in ['Takeaway', 'Dining']:
                    model_key = (item, order_type)
                    if model_key in models:
                        prediction = models[model_key].predict(pred_features)[0]
                        if prediction > 0:
                            all_predictions.append({
                                'Category': category,
                                'Item': item,
                                'Type_of_Order': order_type,
                                'Predicted_Orders': max(0, round(prediction)),
                                'Date': pred_date.strftime('%Y-%m-%d')
                            })
    
    return pd.DataFrame(all_predictions)

# Function to clean expired cache entries
def clean_cache():
    current_time = datetime.now()
    keys_to_delete = [
        key for key, value in cache.items()
        if value['expiration'] < current_time
    ]
    for key in keys_to_delete:
        del cache[key]

# Endpoints
@app.get("/last_date")
async def get_last_date():
    """Get the most recent date in the dataset"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    last_date = df['Date'].max().strftime('%Y-%m-%d')
    return {"last_date": last_date}

@app.post("/reload-data")
async def reload_data():
    """Manually reload data from CSV file"""
    try:
        load_data()
        return {"status": "Data reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload data: {str(e)}")

@app.get("/menu-items", response_model=MenuResponse)
async def get_menu_items_endpoint():
    """Get all menu items grouped by category"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Get menu items grouped by category
    menu_dict = get_menu_items(df)
    
    # Convert numpy arrays to lists for JSON serialization
    menu_dict = {k: v.tolist() for k, v in menu_dict.items()}
    
    # Count total items
    total_items = sum(len(items) for items in menu_dict.values())
    
    return MenuResponse(
        categories=menu_dict,
        total_items=total_items
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get predictions for a specific area and date range with caching"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Generate a cache key
    cache_key = f"{request.area}_{request.target_date}_{request.range}"
    
    # Check the cache
    if cache_key in cache:
        cached_response = cache[cache_key]
        if cached_response['expiration'] > datetime.now():
            logger.info("Cache hit for key: %s", cache_key)
            return cached_response['data']
        else:
            # Expired cache
            del cache[cache_key]
    
    logger.info("Cache miss for key: %s", cache_key)
    
    # Perform the prediction
    predictions_df = predict_for_area_and_range(
        df, 
        request.area, 
        request.target_date, 
        request.range
    )
    
    if predictions_df is None:
        raise HTTPException(status_code=404, detail=f"No data available for area: {request.area}")
    
    # Calculate date range
    date_range = get_date_range(request.target_date, request.range)
    date_range_dict = {
        "start_date": date_range[0].strftime('%Y-%m-%d'),
        "end_date": date_range[-1].strftime('%Y-%m-%d')
    }
    
    # Calculate daily totals
    daily_totals = predictions_df.groupby('Date')['Predicted_Orders'].sum().to_dict()
    daily_totals = {str(k): int(v) for k, v in daily_totals.items()}
    
    # Convert predictions to response format
    predictions_list = predictions_df.to_dict('records')
    total_orders = int(predictions_df['Predicted_Orders'].sum())
    
    response = PredictionResponse(
        predictions=predictions_list,
        total_orders=total_orders,
        area=request.area,
        date_range=date_range_dict,
        daily_totals=daily_totals
    )
    
    # Store response in cache
    cache[cache_key] = {
        "data": response,
        "expiration": datetime.now() + timedelta(hours=CACHE_EXPIRATION_HOURS)
    }
    
    return response
# Set up scheduler to reload data daily
@app.on_event("startup")
async def startup_event():
    # Load initial data
    load_data()
    
    # Schedule data reload
    scheduler.add_job(
        load_data,
        trigger=CronTrigger(hour=0),  # Run at midnight
        id='reload_data'
    )
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)