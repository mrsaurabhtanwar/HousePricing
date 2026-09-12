from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="House Predication API")

class HOUSEInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: Literal[0, 1]
    view: Literal[0, 1]
    condition: Literal[0, 1, 2, 3, 4, 5]
    grade: int
    sqft_above: int
    has_basement: Literal[0, 1]
    sqft_basement: int
    is_renovated: Literal[0, 1]
    house_age: int
    sqft_living15: int
    sqft_lot15: int
    sale_year: int
    sale_month: int
    lat: float
    long: float
    
    

@app.get("/")
def home():
    return{
        "msg": "API is running",
        "info": "/docs"
    }
    
    
@app.post("/predict-price")
def predict_house_price():
    return{
        "msg": "your predication has been successful."
    }