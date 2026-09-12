import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df.copy()
    
    df_model["date"] = pd.to_datetime(df_model["date"])
    df_model["sale_year"] = df_model["date"].dt.year
    df_model["sale_month"] = df_model["date"].dt.month
    
    df_model["house_age"] = df_model["sale_year"] - df_model["yr_built"]
    df_model["is_renovated"] = (df_model["yr_renovated"] > 0).astype(int)
    df_model["has_basement"] = (df_model["sqft_basement"] > 0).astype(int)
    
    if "price" in df_model.columns:
        df_model["log_price"] = np.log(df_model["price"])
        df_model = df_model.drop(columns=["price"])
        
    skewed_cols = [
        "sqft_living", "sqft_lot", "sqft_above",
        "sqft_living15", "sqft_lot15", "sqft_basement"
    ]
    for col in skewed_cols:
        if col in df_model.columns:
            df_model[f"log_{col}"] = np.log1p(df_model[col])
            
    df_model["year_since_renov"] = np.where(
        df_model["is_renovated"] == 1,
        df_model["sale_year"] - df_model["yr_renovated"],
        0
    )
    

    df_model["living_to_lot_ratio"] = (df_model["sqft_living"] / df_model["sqft_lot"]).round(6)
    df_model["sqft_living_diff_from_neighbors"] = df_model["sqft_living"] - df_model["sqft_living15"]
    
    if "zipcode" in df_model.columns:
        df_model["zipcode"] = df_model["zipcode"].astype(str)
        df_model = pd.get_dummies(df_model, columns=["zipcode"], drop_first=True, dtype=int)
        
    df_model = df_model.drop(columns=["id", "date", "yr_built", "yr_renovated"] + skewed_cols)
    
    return df_model
    
    
    
    