import os

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

from backend.schema import HouseInput, PredictionResponse

load_dotenv()

DB_URL = str(os.getenv("DATABASE_URL"))

engine = create_engine(DB_URL)
sessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class HousingDataTable(Base):
    __tablename__ = "Housing_data_table"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Float, nullable=False)
    sqft_living = Column(Integer, nullable=False)
    sqft_lot = Column(Integer, nullable=False)
    floors = Column(Float, nullable=False)
    waterfront = Column(Integer, nullable=False, default=0)
    view = Column(Integer, nullable=False, default=0)
    condition = Column(Integer, nullable=False, default=3)
    grade = Column(Integer, nullable=False, default=7)
    sqft_above = Column(Integer, nullable=False)
    sqft_basement = Column(Integer, nullable=False, default=0)
    yr_built = Column(Integer, nullable=False)
    yr_renovated = Column(Integer, nullable=False, default=0)
    zipcode = Column(String(10), nullable=False)
    lat = Column(Float, nullable=False)
    long = Column(Float, nullable=False)
    sqft_living15 = Column(Integer, nullable=False)
    sqft_lot15 = Column(Integer, nullable=False)
    pred_price = Column(Float, nullable=False)
    create_at = Column(DateTime, default=datetime.now)
    

Base.metadata.create_all(bind=engine)

def get_db():
    db = sessionLocal()
    try:
        if db:
            yield db
    finally:
        db.close()
        
        
def add_housing_data(db: Session, UserData: HouseInput, Pred_response: float):
    try:
        raw_data = HousingDataTable(
            bedrooms=UserData.bedrooms,
            bathrooms=UserData.bathrooms,
            sqft_living=UserData.sqft_living,
            sqft_lot=UserData.sqft_lot,
            floors=UserData.floors,
            waterfront=UserData.waterfront,
            view=UserData.view,
            condition=UserData.condition,
            grade=UserData.grade,
            sqft_above=UserData.sqft_above,
            sqft_basement=UserData.sqft_basement,
            yr_built=UserData.yr_built,
            yr_renovated=UserData.yr_renovated,
            zipcode=UserData.zipcode,
            lat=UserData.lat,
            long=UserData.long,
            sqft_living15=UserData.sqft_living15,
            sqft_lot15=UserData.sqft_lot15,
            pred_price=Pred_response,
        )
        db.add(raw_data)
        db.commit()
        db.refresh(raw_data)
        return{
            "msg": "Data added into DB successfully."
        }
    except Exception as e:
        raise RuntimeError(f"Unable to add data into DB {e}")

0