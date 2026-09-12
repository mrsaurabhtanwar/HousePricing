import requests
import streamlit as st

st.set_page_config(page_title="House Valuation Engine", layout="wide")

st.title("King County House Valuation Engine")

API_URL = "http://localhost:8000/predict-price"

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Property Basics")
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=15, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=10.0, value=2.0, step=0.25)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.5, step=0.5)
    grade = st.slider("Construction Grade (1-13)", min_value=1, max_value=13, value=8)
    condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
    waterfront = st.selectbox("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=0)
    
with col2:
    st.subheader("Square Footage & Age")
    sqft_living = st.number_input("Living Area (sqft)", min_value=200, max_value=15000, value=2100, step=50)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=300, max_value=1000000, value=7500, step=100)
    sqft_above = st.number_input("Above Ground (sqft)", min_value=200, max_value=15000, value=1600, step=50)
    sqft_basement = st.number_input("Basement (sqft)", min_value=0, max_value=10000, value=500, step=50)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1985)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2026, value=0)
    
with col3:
    st.subheader("Location & Neighbors")
    zipcode = st.text_input("Zipcode", value="98103")
    lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.6700, format="%.4f")
    long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.3500, format="%.4f")
    sqft_living15 = st.number_input("Nearest 15 Neighbors Living (sqft)", min_value=200, max_value=15000, value=1900, step=50)
    sqft_lot15 = st.number_input("Nearest 15 Neighbors Lot (sqft)", min_value=300, max_value=1000000, value=7200, step=100)


if st.button("Calculate Valuation", type="primary"):
    payload = {
        "bedrooms": int(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": float(floors),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "grade": int(grade),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "yr_built": int(yr_built),
        "yr_renovated": int(yr_renovated),
        "zipcode": str(zipcode).strip(),
        "lat": float(lat),
        "long": float(long),
        "sqft_living15": int(sqft_living15),
        "sqft_lot15": int(sqft_lot15),
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.success(f"Estimated Valuation: ${data['predicted_price_dollars']:,.2f} USD")
            st.info(f"Valuation Range (±10%): ${data['valuation_low_estimate']:,.2f} — ${data['valuation_high_estimate']:,.2f} USD")
        else:
           st.error(f"API Error ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI server. Make sure Uvicorn is running on port 8000.")