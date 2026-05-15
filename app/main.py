from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(
    title="STRATOS Prediction API",
    description="Real-time heat warning predictions for Misamis Oriental.",
    version="1.0.0"
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# We will try to load the model at startup, but won't crash if it's missing initially.
# It will be generated when the user runs the Colab notebook.
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Please run the Colab notebook first to generate it.")
except Exception as e:
    print(f"Error loading model: {e}")

class WeatherPayload(BaseModel):
    temperature_2m: float
    relative_humidity_2m: float
    solar_radiation: float
    wind_speed_10m: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the STRATOS Prediction API. Use the /predict endpoint."}

@app.post("/predict")
def predict_extreme_heat(payload: WeatherPayload):
    global model
    if model is None:
        # Check again in case it was added while running
        try:
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
            else:
                raise HTTPException(status_code=503, detail="Model not loaded. Please ensure model.pkl exists in the app directory.")
        except Exception:
            raise HTTPException(status_code=503, detail="Model not loaded. Please ensure model.pkl exists in the app directory.")

    # Convert payload to DataFrame to match expected input format
    input_data = pd.DataFrame([{
        "temperature_2m": payload.temperature_2m,
        "relative_humidity_2m": payload.relative_humidity_2m,
        "solar_radiation": payload.solar_radiation,
        "wind_speed_10m": payload.wind_speed_10m
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    
    # We output 1 if extreme heat warning, else 0.
    is_extreme = bool(prediction == 1)
    
    return {
        "extreme_heat_warning": is_extreme,
        "input_features": payload.dict()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
