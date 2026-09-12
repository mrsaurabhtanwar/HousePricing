from pydantic import BaseModel, Field


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