import numpy as np
import pandas as pd
import pytest

from mlpipe.features import feature_engineering
from backend.schema import HouseInput
from backend.main import build_feature_dataframe


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": 1,
            "date": "20141013T000000",
            "price": 221900.0,
            "bedrooms": 3,
            "bathrooms": 1.0,
            "sqft_living": 1180,
            "sqft_lot": 5650,
            "floors": 1.0,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "grade": 7,
            "sqft_above": 1180,
            "sqft_basement": 0,
            "yr_built": 1955,
            "yr_renovated": 0,
            "zipcode": 98178,
            "lat": 47.5112,
            "long": -122.257,
            "sqft_living15": 1340,
            "sqft_lot15": 5650,
        },
        {
            "id": 2,
            "date": "20150218T000000",
            "price": 538000.0,
            "bedrooms": 3,
            "bathrooms": 2.25,
            "sqft_living": 2570,
            "sqft_lot": 7242,
            "floors": 2.0,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "grade": 7,
            "sqft_above": 2170,
            "sqft_basement": 400,
            "yr_built": 1951,
            "yr_renovated": 1991,
            "zipcode": 98125,
            "lat": 47.7210,
            "long": -122.319,
            "sqft_living15": 1690,
            "sqft_lot15": 7639,
        },
    ])


def test_feature_engineering_columns_and_transforms(sample_raw_data: pd.DataFrame):
    df_transformed = feature_engineering(sample_raw_data)

    for dropped_col in ["id", "date", "yr_built", "yr_renovated", "sqft_living", "sqft_lot", "price"]:
        assert dropped_col not in df_transformed.columns

    expected_new_cols = [
        "log_price",
        "sale_year",
        "sale_month",
        "house_age",
        "is_renovated",
        "has_basement",
        "year_since_renov",
        "living_to_lot_ratio",
        "sqft_living_diff_from_neighbors",
        "log_sqft_living",
        "log_sqft_lot",
        "log_sqft_above",
        "log_sqft_basement",
    ]
    for col in expected_new_cols:
        assert col in df_transformed.columns

    assert df_transformed.loc[0, "is_renovated"] == 0
    assert df_transformed.loc[0, "year_since_renov"] == 0
    assert df_transformed.loc[0, "has_basement"] == 0

    assert df_transformed.loc[1, "is_renovated"] == 1
    assert df_transformed.loc[1, "year_since_renov"] == 24
    assert df_transformed.loc[1, "has_basement"] == 1


def test_build_feature_dataframe():
    house = HouseInput(
        bedrooms=3,
        bathrooms=2.0,
        sqft_living=2100,
        sqft_lot=7500,
        floors=1.5,
        waterfront=0,
        view=0,
        condition=3,
        grade=8,
        sqft_above=1600,
        sqft_basement=500,
        yr_built=1985,
        yr_renovated=0,
        zipcode="98103",
        lat=47.67,
        long=-122.35,
        sqft_living15=1900,
        sqft_lot15=7200,
    )

    feature_names = [
        "bedrooms",
        "bathrooms",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "lat",
        "long",
        "sale_year",
        "sale_month",
        "house_age",
        "is_renovated",
        "has_basement",
        "log_sqft_living",
        "log_sqft_lot",
        "log_sqft_above",
        "log_sqft_living15",
        "log_sqft_lot15",
        "log_sqft_basement",
        "year_since_renov",
        "living_to_lot_ratio",
        "sqft_living_diff_from_neighbors",
        "zipcode_98103",
        "zipcode_98125",
    ]

    df = build_feature_dataframe(house, feature_names)

    assert df.shape == (1, len(feature_names))
    assert list(df.columns) == feature_names
    assert df.loc[0, "zipcode_98103"] == 1
    assert df.loc[0, "zipcode_98125"] == 0
    assert df.loc[0, "has_basement"] == 1
    assert df.loc[0, "is_renovated"] == 0
