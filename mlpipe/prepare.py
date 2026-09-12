from pathlib import Path
import pandas as pd

from mlpipe.config import get_project_root, load_config
from mlpipe.features import feature_engineering

def prepare_data():
    root = get_project_root()
    config = load_config()
    
    raw_data_path = root / "data" / "kc_house_data.csv"
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at : {raw_data_path}")
    
    print(f"Reading raw data from : {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    df.loc[df["bedrooms"] == 33, "bedrooms"] = 3
    
    df = df[df["bathrooms"] > 0].copy()
    
    print("Apply feature engineering...")
    df_processed = feature_engineering(df)
    
    output_path = Path(config["data"]["processed_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Processed data successfully saved to : {output_path}.")
    print(f"Shape: {df_processed.shape}")
    
if __name__ == "__main__":
    prepare_data()    
    