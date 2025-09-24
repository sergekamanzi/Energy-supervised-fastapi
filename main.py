import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rwanda Energy Consumption Predictor API",
    description="API for predicting household energy consumption in Rwanda",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for models
model = None
scaler = None
label_encoders = None
feature_names = None
constants = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)



# Initialize models
def initialize_models():
    """Initialize all models and preprocessing objects"""
    global model, scaler, label_encoders, feature_names, constants
    
    try:
        logger.info("Loading model and preprocessing objects...")
        
        # Load preprocessing objects first
        scaler = joblib.load('energy_scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        constants = joblib.load('model_constants.pkl')
        
        # Load model with custom objects to handle compatibility
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
        }
        
        model = load_model('rwanda_energy_model.h5', custom_objects=custom_objects, compile=False)
        
        # Simple compilation
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("✅ All models and preprocessing objects loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        # Set defaults to prevent crashes
        constants = {
            'TARIFF_RATES': {'0-20 kWh': 89, '21-50 kWh': 310, '50+ kWh': 369},
            'INCOME_LEVELS': {'Low': '<100,000 RWF/month', 'Medium': '100,000-500,000 RWF/month', 'High': '>500,000 RWF/month'}
        }

# Initialize on startup
initialize_models()

# Pydantic models for request/response
class ApplianceInput(BaseModel):
    appliance: str
    power: float
    power_unit: str = "W"
    hours: float
    quantity: int
    usage_days_monthly: int = 30

class HouseholdInfo(BaseModel):
    region: str
    income_level: str
    appliances_count: int
    household_size: int
    budget: float

class PredictionRequest(BaseModel):
    appliances: List[ApplianceInput]
    household_info: HouseholdInfo

class ApplianceResult(BaseModel):
    appliance: str
    power_watts: float
    hours_daily: float
    quantity: int
    usage_days_monthly: int
    estimated_kwh: float
    estimated_bill: float
    percentage: float

class PredictionResponse(BaseModel):
    total_kwh: float
    total_bill: float
    tariff_bracket: str
    breakdown: List[ApplianceResult]
    budget_status: str
    budget_difference: float
    status: str
    message: str

# Utility functions
def convert_power_units(power_value: float, input_unit: str, output_unit: str = 'W') -> float:
    """Convert power between different units"""
    if input_unit.upper() == 'KW':
        watts = power_value * 1000
    elif input_unit.upper() == 'MW':
        watts = power_value * 1000000
    else:
        watts = power_value

    if output_unit.upper() == 'KW':
        return watts / 1000
    elif output_unit.upper() == 'MW':
        return watts / 1000000
    else:
        return watts

def calculate_tariff_bracket(total_kwh: float) -> str:
    """Determine tariff bracket based on monthly consumption"""
    if total_kwh <= 20:
        return '0-20 kWh'
    elif total_kwh <= 50:
        return '21-50 kWh'
    else:
        return '50+ kWh'

def calculate_estimated_bill(total_kwh: float) -> float:
    """Calculate estimated bill based on Rwanda tariff structure"""
    tariff_rates = constants['TARIFF_RATES']
    
    if total_kwh <= 20:
        return total_kwh * tariff_rates['0-20 kWh']
    elif total_kwh <= 50:
        return (20 * tariff_rates['0-20 kWh'] +
                (total_kwh - 20) * tariff_rates['21-50 kWh'])
    else:
        return (20 * tariff_rates['0-20 kWh'] +
                30 * tariff_rates['21-50 kWh'] +
                (total_kwh - 50) * tariff_rates['50+ kWh'])

def simple_energy_calculation(appliances_list: List[ApplianceInput]) -> Dict:
    """Simple energy calculation as fallback if model fails"""
    total_kwh = 0
    breakdown = []
    
    for appliance in appliances_list:
        # Convert power to Watts
        power_watts = convert_power_units(appliance.power, appliance.power_unit, 'W')
        
        # Calculate monthly energy consumption
        monthly_kwh = (power_watts * appliance.hours * appliance.quantity * appliance.usage_days_monthly) / 1000
        total_kwh += monthly_kwh
        
        breakdown.append({
            "appliance": appliance.appliance,
            "power_watts": power_watts,
            "hours_daily": appliance.hours,
            "quantity": appliance.quantity,
            "usage_days_monthly": appliance.usage_days_monthly,
            "estimated_kwh": monthly_kwh,
            "estimated_bill": 0,  # Will be calculated later
            "percentage": 0
        })
    
    # Calculate bill and percentages
    total_bill = calculate_estimated_bill(total_kwh)
    
    for item in breakdown:
        item['estimated_bill'] = (item['estimated_kwh'] / total_kwh) * total_bill if total_kwh > 0 else 0
        item['percentage'] = (item['estimated_kwh'] / total_kwh) * 100 if total_kwh > 0 else 0
    
    return {
        "total_kwh": total_kwh,
        "total_bill": total_bill,
        "breakdown": breakdown
    }

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Rwanda Energy Consumption Predictor API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "docs": "/docs",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health_check():
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded", 
        "model_loaded": model_loaded,
        "message": "Model loaded successfully" if model_loaded else "Using fallback calculation",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/info")
async def model_info():
    return {
        "model_name": "Rwanda Energy Consumption Predictor",
        "version": "1.0.0",
        "tariff_rates": constants['TARIFF_RATES'],
        "income_levels": constants['INCOME_LEVELS'],
        "status": "operational"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_energy_consumption(request: PredictionRequest):
    """
    Predict energy consumption for a household
    """
    try:
        logger.info(f"Received prediction request for {len(request.appliances)} appliances")
        
        # Use model prediction if available, otherwise use fallback
        if model is not None and scaler is not None:
            try:
                # Create input DataFrame
                appliance_data = []
                for appliance in request.appliances:
                    power_watts = convert_power_units(appliance.power, appliance.power_unit, 'W')
                    daily_energy_kwh = (power_watts * appliance.hours * appliance.quantity) / 1000
                    
                    appliance_data.append({
                        'Appliance': appliance.appliance,
                        'Power_Watts': power_watts,
                        'Usage_Hours_Daily': appliance.hours,
                        'Quantity': appliance.quantity,
                        'Region': request.household_info.region,
                        'Income_Level': request.household_info.income_level,
                        'Appliances_Count': request.household_info.appliances_count,
                        'Usage_Days_Monthly': appliance.usage_days_monthly,
                        'Household_Size': request.household_info.household_size,
                        'Daily_Energy_kWh': daily_energy_kwh,
                        'Monthly_Usage_Factor': appliance.usage_days_monthly / 30,
                        'Appliance_Load': power_watts * appliance.quantity,
                        'Energy_Intensity': 0
                    })

                input_df = pd.DataFrame(appliance_data)
                
                # Ensure all columns are present
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df[feature_names]
                
                # Scale features and predict
                input_scaled = scaler.transform(input_df)
                predictions = model.predict(input_scaled)
                
                # Handle predictions
                if isinstance(predictions, list):
                    kwh_pred = predictions[0].flatten()
                else:
                    kwh_pred = predictions.flatten()
                
                total_kwh = float(np.sum(kwh_pred))
                message = "Prediction completed using AI model"
                
            except Exception as model_error:
                logger.warning(f"Model prediction failed, using fallback: {model_error}")
                # Fallback to simple calculation
                result = simple_energy_calculation(request.appliances)
                total_kwh = result["total_kwh"]
                breakdown_data = result["breakdown"]
                message = "Prediction completed using fallback calculation"
        else:
            # Use simple calculation if model not loaded
            result = simple_energy_calculation(request.appliances)
            total_kwh = result["total_kwh"]
            breakdown_data = result["breakdown"]
            message = "Prediction completed using simple calculation (model not loaded)"
        
        # Calculate final results
        predicted_tariff = calculate_tariff_bracket(total_kwh)
        total_bill = calculate_estimated_bill(total_kwh)
        
        # Budget analysis
        budget_difference = total_bill - request.household_info.budget
        budget_status = "within_budget" if budget_difference <= 0 else "over_budget"
        
        # Create breakdown if not already created
        if 'breakdown_data' not in locals():
            breakdown_data = []
            for i, appliance in enumerate(request.appliances):
                if i < len(kwh_pred):
                    appliance_kwh = float(kwh_pred[i])
                else:
                    # Fallback calculation
                    power_watts = convert_power_units(appliance.power, appliance.power_unit, 'W')
                    appliance_kwh = (power_watts * appliance.hours * appliance.quantity * appliance.usage_days_monthly) / 1000
                
                appliance_bill = (appliance_kwh / total_kwh) * total_bill if total_kwh > 0 else 0
                
                breakdown_data.append({
                    "appliance": appliance.appliance,
                    "power_watts": convert_power_units(appliance.power, appliance.power_unit, 'W'),
                    "hours_daily": appliance.hours,
                    "quantity": appliance.quantity,
                    "usage_days_monthly": appliance.usage_days_monthly,
                    "estimated_kwh": appliance_kwh,
                    "estimated_bill": appliance_bill,
                    "percentage": (appliance_kwh / total_kwh) * 100 if total_kwh > 0 else 0
                })
        
        # Sort by consumption
        breakdown_data.sort(key=lambda x: x['estimated_kwh'], reverse=True)
        
        # Convert to Pydantic models
        breakdown = [ApplianceResult(**item) for item in breakdown_data]
        
        return PredictionResponse(
            total_kwh=total_kwh,
            total_bill=total_bill,
            tariff_bracket=predicted_tariff,
            breakdown=breakdown,
            budget_status=budget_status,
            budget_difference=budget_difference,
            status="success",
            message=message
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Error handler that returns proper JSONResponse
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")