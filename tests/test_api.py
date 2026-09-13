import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from database.housing_database import Base, get_db, HousingDataTable


@pytest.fixture
def client_with_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client, TestingSessionLocal

    app.dependency_overrides.clear()


@pytest.fixture
def valid_house_payload():
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2100,
        "sqft_lot": 7500,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 8,
        "sqft_above": 1600,
        "sqft_basement": 500,
        "yr_built": 1985,
        "yr_renovated": 0,
        "zipcode": "98103",
        "lat": 47.67,
        "long": -122.35,
        "sqft_living15": 1900,
        "sqft_lot15": 7200,
    }


def test_home_endpoint(client_with_db):
    client, _ = client_with_db
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["msg"] == "API is running"
    assert data["model_loaded"] is True
    assert data["model_type"] == "LightGBM"


def test_predict_price_success(client_with_db, valid_house_payload):
    client, SessionLocal = client_with_db
    response = client.post("/predict-price", json=valid_house_payload)

    assert response.status_code == 200
    data = response.json()

    assert "predicted_price_dollars" in data
    assert "valuation_low_estimate" in data
    assert "valuation_high_estimate" in data
    assert data["currency"] == "USD"

    predicted = data["predicted_price_dollars"]
    assert predicted > 0
    assert data["valuation_low_estimate"] < predicted < data["valuation_high_estimate"]

    with SessionLocal() as db:
        rows = db.query(HousingDataTable).all()
        assert len(rows) == 1
        assert rows[0].zipcode == "98103"
        assert rows[0].pred_price == pytest.approx(predicted, 0.01)


def test_predict_price_validation_error_invalid_field(client_with_db, valid_house_payload):
    client, _ = client_with_db
    invalid_payload = valid_house_payload.copy()
    invalid_payload["bedrooms"] = 0 

    response = client.post("/predict-price", json=invalid_payload)
    assert response.status_code == 422


def test_predict_price_validation_error_missing_field(client_with_db, valid_house_payload):
    client, _ = client_with_db
    incomplete_payload = valid_house_payload.copy()
    del incomplete_payload["zipcode"]

    response = client.post("/predict-price", json=incomplete_payload)
    assert response.status_code == 422
