from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
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

# Pydantic models
class PredictionRequest(BaseModel):
    area: str
    target_date: str

class PredictionItem(BaseModel):
    Category: str
    Item: str
    Type_of_Order: str
    Predicted_Orders: int

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    total_orders: int
    area: str
    target_date: str

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

def predict_for_area(df, area: str, target_date: str) -> Optional[pd.DataFrame]:
    area_data = df[df['Area'] == area].copy()
    
    if len(area_data) == 0:
        return None
    
    menu_items = get_menu_items(df)
    models = train_area_models(area_data)
    
    try:
        pred_date = pd.to_datetime(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    pred_features = pd.DataFrame({
        'Year': [pred_date.year],
        'Month': [pred_date.month],
        'DayOfWeek': [pred_date.dayofweek],
        'DayOfMonth': [pred_date.day]
    })
    
    predictions = []
    for category, items in menu_items.items():
        for item in items:
            for order_type in ['Takeaway', 'Dining']:
                model_key = (item, order_type)
                if model_key in models:
                    prediction = models[model_key].predict(pred_features)[0]
                    if prediction > 0:
                        predictions.append({
                            'Category': category,
                            'Item': item,
                            'Type_of_Order': order_type,
                            'Predicted_Orders': max(0, round(prediction))
                        })
    
    return pd.DataFrame(predictions)

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
    """Get predictions for a specific area and date"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    predictions_df = predict_for_area(df, request.area, request.target_date)
    
    if predictions_df is None:
        raise HTTPException(status_code=404, detail=f"No data available for area: {request.area}")
    
    # Convert predictions to response format
    predictions_list = predictions_df.to_dict('records')
    total_orders = int(predictions_df['Predicted_Orders'].sum())
    
    return PredictionResponse(
        predictions=predictions_list,
        total_orders=total_orders,
        area=request.area,
        target_date=request.target_date
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)