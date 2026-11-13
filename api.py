from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Lung Cancer Risk DSS API",
    description="Backend engine for predicting lung cancer risk based on patient data.",
    version="1.0"
)

try:
    pipeline = joblib.load('xgboost_pipeline.joblib')
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

# Define the Data Schema using Pydantic
class PatientData(BaseModel):
    age: int
    gender: int
    education_years: int
    income_level: int
    smoker: int
    smoking_years: int
    cigarettes_per_day: int
    pack_years: float
    passive_smoking: int
    air_pollution_index: int
    occupational_exposure: int
    radon_exposure: int
    family_history_cancer: int
    asthma: int
    previous_tb: int
    chronic_cough: int
    chest_pain: int
    shortness_of_breath: int
    fatigue: int
    bmi: float
    oxygen_saturation: float
    exercise_hours_per_week: float
    diet_quality: int
    alcohol_units_per_week: float
    healthcare_access: int

# Create the prediction endpoint
@app.post("/predict")
def predict_risk(patient: PatientData):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded.")
   
    # Convert the validated incoming data into a single-row Pandas DataFrame
    input_df = pd.DataFrame([patient.model_dump()])
   
    try:
        # The pipeline automatically applies the OrdinalEncoder, then XGBoost makes the prediction
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[0][1] # Probability of class 1 (High Risk)
       
        # Return the assessment as a JSON response
        return {
            "prediction": int(prediction[0]),
            "risk_probability": float(probability),
            "assessment": "High Risk" if prediction[0] == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# A simple health-check endpoint
@app.get("/")
def read_root():
    return {"message": "DSS API is running. Send a POST request to /predict."}
