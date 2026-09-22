from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI()


# Load the NEW model
model = joblib.load("used_car_price_model.pkl")


class CarDetails(BaseModel):
    Brand: str
    model: str
    Year: int
    kmDriven: int
    Transmission: str
    Owner: str
    FuelType: str


# Open frontend
@app.get("/")
def home():
    return FileResponse("static/index.html")


# Prediction API
@app.post("/predict")
def predict_price(car: CarDetails):

    car_data = pd.DataFrame([{
        "Brand": car.Brand,
        "model": car.model,
        "Year": car.Year,
        "kmDriven": car.kmDriven,
        "Transmission": car.Transmission,
        "Owner": car.Owner,
        "FuelType": car.FuelType
    }])

    prediction = model.predict(car_data)[0]

    return {
        "predicted_price": round(prediction)
    }