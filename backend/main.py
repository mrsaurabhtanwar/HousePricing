from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


class HouseInput(BaseModel):
    bedrooms: int = Field(..., ge=1, le=15, example=3)
    bathrooms: float = Field(..., ge=0.5, le=10.0, example=2.0)
    sqft_living: int = Field(..., ge=200, le=15000, example=2100)
    sqft_lot: int = Field(..., ge=300, le=1000000, example=7500)
    floors: float = Field(..., ge=1.0, le=4.0, example=1.5)
    waterfront: int = Field(0, ge=0, le=1, example=0)
    view: int = Field(0, ge=0, le=4, example=0)
    condition: int = Field(3, ge=1, le=5, example=3)
    grade: int = Field(7, ge=1, le=13, example=8)
    sqft_above: int = Field(..., ge=200, le=15000, example=1600)
    sqft_basement: int = Field(0, ge=0, le=10000, example=500)
    yr_built: int = Field(..., ge=1900, le=2026, example=1985)
    yr_renovated: int = Field(0, ge=0, le=2026, example=0)
    zipcode: str = Field(..., example="98103")
    lat: float = Field(..., ge=47.0, le=48.0, example=47.67)
    long: float = Field(..., ge=-123.0, le=-121.0, example=-122.35)
    sqft_living15: int = Field(..., ge=200, le=15000, example=1900)
    sqft_lot15: int = Field(..., ge=300, le=1000000, example=7200)
    

class PredictionResponse(BaseModel):
    predicted_price_dollars: float
    valuation_low_estimate: float
    valuation_high_estimate: float
    currency: str = "USD"
    
    
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).resolve().parent.parent / "models" / "lightgbm_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    model_state["model"] = joblib.load(model_path)
    model_state["feature_names"] = model_state["model"].feature_name_
    yield
    model_state.clear()
    
app = FastAPI(
    title="House Predication API",
    version="1.0.0",
    lifespan=lifespan
)


def build_feature_dataframe(house: HouseInput, feature_names: list[str]) -> pd.DataFrame:
    current_date = datetime.now()
    sale_year = current_date.year
    sale_month = current_date.month
    
    house_age = sale_year - house.yr_built
    is_renovated = 1 if house.yr_renovated > 0 else 0
    has_basement = 1 if house.sqft_basement > 0 else 0
    year_since_renov = (sale_year - house.yr_renovated) if is_renovated else 0
    living_to_lot_ratio = round(house.sqft_living / house.sqft_lot, 6)
    sqft_living_diff = house.sqft_living - house.sqft_living15
    
    row = {
        "bedrooms": house.bedrooms,
        "bathrooms": house.bathrooms,
        "floors": house.floors,
        "waterfront": house.waterfront,
        "view": house.view,
        "condition": house.condition,
        "grade": house.grade,
        "lat": house.lat,
        "long": house.long,
        "sale_year": sale_year,
        "sale_month": sale_month,
        "house_age": house_age,
        "is_renovated": is_renovated,
        "has_basement": has_basement,
        "log_sqft_living": np.log1p(house.sqft_living),
        "log_sqft_lot": np.log1p(house.sqft_lot),
        "log_sqft_above": np.log1p(house.sqft_above),
        "log_sqft_living15": np.log1p(house.sqft_living15),
        "log_sqft_lot15": np.log1p(house.sqft_lot15),
        "log_sqft_basement": np.log1p(house.sqft_basement),
        "year_since_renov": year_since_renov,
        "living_to_lot_ratio": living_to_lot_ratio,
        "sqft_living_diff_from_neighbors": sqft_living_diff,
    }

    target_zip = f"zipcode_{str(house.zipcode).strip()}"
    for col in feature_names:
        if col.startswith("zipcode_"):
            row[col] = 1 if col == target_zip else 0
            
    df = pd.DataFrame([row])[feature_names]
    return df  


@app.get("/")
def home():
    return{
        "msg": "API is running",
        "model_loaded": "model" in model_state,
        "model_type": "LightGBM",
        "info": "/docs"
    }
    
    
@app.post("/predict-price", response_model=PredictionResponse)
def predict_house_price(house: HouseInput):
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    model = model_state["model"]
    feature_name = model_state["feature_names"]
    
    df = build_feature_dataframe(house, feature_name)
    log_prediction = model.predict(df)[0]
    
    predicted_price = float(np.exp(log_prediction))
    
    low_estimate = round(predicted_price * 0.90, 2)
    high_estimate = round(predicted_price * 1.10, 2)  
     
    return PredictionResponse(
        predicted_price_dollars=round(predicted_price, 2),
        valuation_low_estimate=low_estimate,
        valuation_high_estimate=high_estimate,
    )