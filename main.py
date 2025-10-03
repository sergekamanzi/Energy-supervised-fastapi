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
    description="API for predicting household energy consumption in Rwanda with detailed appliance analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for models
model = None
scaler = None
label_encoders = None
feature_names = None
X_train_columns = None

# Rwanda Electricity Tariff Rates (October 2025)
TARIFF_RATES = {
    '0-20 kWh': 89,
    '21-50 kWh': 310,
    '50+ kWh': 369
}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def initialize_models():
    """Initialize all models and preprocessing objects"""
    global model, scaler, label_encoders, feature_names, X_train_columns
    
    try:
        logger.info("Loading model and preprocessing objects...")
        
        # Load preprocessing objects
        scaler = joblib.load('energy_scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        X_train_columns = joblib.load('X_train_columns.pkl')  # This should contain the training data structure
        
        logger.info(" Preprocessing objects loaded successfully!")
        
        # Try to load TensorFlow model
        try:
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
            }
            model = load_model('rwanda_energy_model.h5', custom_objects=custom_objects, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info(" TensorFlow model loaded successfully!")
        except Exception as e:
            logger.warning(f"TensorFlow model not loaded, using scikit-learn: {e}")
            # Fallback to scikit-learn model
            try:
                model = joblib.load('energy_model.pkl')
                logger.info(" Scikit-learn model loaded successfully!")
            except Exception as e2:
                logger.warning(f"Scikit-learn model also failed: {e2}")
                model = None
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        model = None
        scaler = None
        label_encoders = None

# Initialize on startup
initialize_models()

# Pydantic models for request/response
class ApplianceInput(BaseModel):
    appliance: str
    power: float
    power_unit: str = "W"
    hours: float
    quantity: int = 1
    usage_days_monthly: int = 30

class HouseholdInfo(BaseModel):
    region: str
    income_level: str
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

class RecommendationItem(BaseModel):
    category: str
    message: str

class PredictionResponse(BaseModel):
    total_kwh: float
    total_bill: float
    tariff_bracket: str
    breakdown: List[ApplianceResult]
    budget_status: str
    budget_difference: float
    recommendations: List[RecommendationItem]
    status: str
    message: str

# Utility functions (from your Colab script)
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
    if total_kwh <= 20:
        return total_kwh * TARIFF_RATES['0-20 kWh']
    elif total_kwh <= 50:
        return (20 * TARIFF_RATES['0-20 kWh'] +
                (total_kwh - 20) * TARIFF_RATES['21-50 kWh'])
    else:
        return (20 * TARIFF_RATES['0-20 kWh'] +
                30 * TARIFF_RATES['21-50 kWh'] +
                (total_kwh - 50) * TARIFF_RATES['50+ kWh'])

def preprocess_user_input(appliances_list: List[ApplianceInput], household_info: HouseholdInfo):
    """Preprocess user input for ML model prediction (from Colab script)"""
    
    appliance_data = []
    for appliance in appliances_list:
        # Convert power to Watts
        power_watts = convert_power_units(
            appliance.power,
            appliance.power_unit,
            'W'
        )

        appliance_data.append({
            'Appliance': appliance.appliance,
            'Power_Watts': power_watts,
            'Usage_Hours_Daily': appliance.hours,
            'Quantity': appliance.quantity,
            'Usage_Days_Monthly': appliance.usage_days_monthly,
            'Region': household_info.region,
            'Income_Level': household_info.income_level,
            'Household_Size': household_info.household_size
        })

    # Create DataFrame
    input_df = pd.DataFrame(appliance_data)
    
    return input_df, appliance_data

def handle_unseen_appliances(input_df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Handle unseen appliances by mapping to known ones"""
    if label_encoders is None:
        return input_df
        
    known_appliances = set(label_encoders['Appliance'].classes_)
    for idx, appliance_name in enumerate(input_df['Appliance']):
        if appliance_name not in known_appliances:
            # Map to default appliance
            most_common = 'LED Lamp'
            logger.info(f"Mapping '{appliance_name}' to '{most_common}'")
            input_df.loc[idx, 'Appliance'] = most_common
    return input_df

def encode_categorical_features(input_df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Encode categorical features using label encoders"""
    if label_encoders is None:
        return input_df
        
    for col in ['Appliance', 'Region', 'Income_Level']:
        if col in input_df.columns and col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    return input_df

def predict_with_model(appliances_list: List[ApplianceInput], household_info: HouseholdInfo):
    """Predict energy consumption using ML model"""
    
    # Preprocess input
    input_df, appliance_data = preprocess_user_input(appliances_list, household_info)
    
    # Handle unseen categories
    if label_encoders is not None:
        input_df = handle_unseen_appliances(input_df, label_encoders)
        input_df = encode_categorical_features(input_df, label_encoders)
    
    # Ensure all required columns are present
    if X_train_columns is not None:
        for col in X_train_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_train_columns]
    
    # Scale features and predict
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
    
    if model is not None:
        # Make predictions
        appliance_predictions = model.predict(input_scaled)
        
        # Convert predictions to flat array
        try:
            predictions_flat = np.array(appliance_predictions).flatten()
        except:
            predictions_flat = []
            for pred in appliance_predictions:
                if hasattr(pred, '__iter__') and not isinstance(pred, str):
                    predictions_flat.extend([float(p) for p in pred])
                else:
                    predictions_flat.append(float(pred))
            predictions_flat = np.array(predictions_flat)
        
        return predictions_flat, appliance_data
    else:
        raise Exception("No model available for prediction")

def simple_energy_calculation(appliances_list: List[ApplianceInput]) -> Dict:
    """Simple energy calculation as fallback"""
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
            "estimated_bill": 0,
            "percentage": 0
        })
    
    return total_kwh, breakdown

def get_smart_recommendations(total_kwh: float, total_bill: float, breakdown: List, budget: float):
    """Generate smart recommendations based on analysis"""
    recommendations = []
    tariff_bracket = calculate_tariff_bracket(total_kwh)
    
    # Budget analysis
    recommendations.append(RecommendationItem(
        category="budget",
        message="BUDGET ANALYSIS"
    ))
    
    if total_bill > budget:
        overspend = total_bill - budget
        recommendations.append(RecommendationItem(
            category="budget",
            message=f"• You're exceeding your budget by {overspend:.0f} RWF"
        ))
        
        # Calculate reduction needed
        if tariff_bracket == '50+ kWh' and calculate_estimated_bill(50) <= budget:
            reduction = total_kwh - 50
            recommendations.append(RecommendationItem(
                category="budget",
                message=f"• Reduce consumption by {reduction:.1f} kWh to reach 21-50 kWh bracket"
            ))
        elif tariff_bracket == '21-50 kWh' and calculate_estimated_bill(20) <= budget:
            reduction = total_kwh - 20
            recommendations.append(RecommendationItem(
                category="budget",
                message=f"• Reduce consumption by {reduction:.1f} kWh to reach 0-20 kWh bracket"
            ))
    else:
        savings = budget - total_bill
        recommendations.append(RecommendationItem(
            category="budget",
            message=f"• You're within budget with {savings:.0f} RWF remaining"
        ))
    
    # Tariff insights
    recommendations.append(RecommendationItem(
        category="tariff",
        message="⚡ TARIFF BRACKET INSIGHTS"
    ))
    recommendations.append(RecommendationItem(
        category="tariff",
        message=f"• Current bracket: {tariff_bracket} ({TARIFF_RATES[tariff_bracket]} RWF/kWh)"
    ))
    
    if tariff_bracket == '0-20 kWh' and total_kwh > 18:
        recommendations.append(RecommendationItem(
            category="tariff",
            message="• Close to 21-50 kWh bracket - stay under 20 kWh"
        ))
    elif tariff_bracket == '21-50 kWh' and total_kwh > 45:
        recommendations.append(RecommendationItem(
            category="tariff",
            message="• Close to 50+ kWh bracket - try to stay under 50 kWh"
        ))
    
    # Top appliance insights
    recommendations.append(RecommendationItem(
        category="appliances",
        message="🔌 TOP ENERGY CONSUMERS"
    ))
    
    top_consumers = sorted(breakdown, key=lambda x: x['estimated_kwh'], reverse=True)[:3]
    for i, appliance in enumerate(top_consumers, 1):
        recommendations.append(RecommendationItem(
            category="appliances",
            message=f"{i}. {appliance['appliance']}: {appliance['estimated_kwh']:.1f} kWh "
                   f"({appliance['percentage']:.1f}%) - {appliance['estimated_bill']:.0f} RWF"
        ))
    
    # Usage optimization tips
    recommendations.append(RecommendationItem(
        category="optimization",
        message=" USAGE OPTIMIZATION TIPS"
    ))
    
    if any(item['hours_daily'] > 8 for item in breakdown):
        recommendations.append(RecommendationItem(
            category="optimization",
            message="• Reduce usage hours for appliances running more than 8 hours daily"
        ))
    
    high_power_items = [item for item in breakdown if item['power_watts'] > 1000]
    if high_power_items:
        recommendations.append(RecommendationItem(
            category="optimization",
            message="• Consider using high-power appliances during off-peak hours"
        ))
    
    high_consumption = [item for item in breakdown if item['estimated_kwh'] > 10]
    if high_consumption:
        recommendations.append(RecommendationItem(
            category="optimization",
            message="• Focus on reducing usage for high-consumption appliances first"
        ))
    
    return recommendations

# API endpoints - Only keeping essential ones
@app.get("/")
async def root():
    return {
        "message": "Rwanda Energy Consumption Predictor API",
        "version": "2.0.0",
        "status": "running",
        "description": "Advanced energy prediction with appliance-level analysis",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health_check():
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded", 
        "model_loaded": model_loaded,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None,
        "message": "Full ML pipeline available" if model_loaded and scaler and label_encoders else "Using fallback calculation",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_energy_consumption(request: PredictionRequest):
    """
    Predict energy consumption for a household with detailed appliance analysis
    
    **Features:**
    - ML-based energy prediction per appliance
    - Rwanda tariff bracket calculation
    - Budget analysis and recommendations
    - Appliance-level consumption breakdown
    - Smart optimization tips
    """
    try:
        logger.info(f"Received prediction request for {len(request.appliances)} appliances")
        
        # Validate input
        if not request.appliances:
            raise HTTPException(status_code=400, detail="At least one appliance is required")
        
        if request.household_info.household_size <= 0:
            raise HTTPException(status_code=400, detail="Household size must be positive")
        
        # Use model prediction if available
        if model is not None and scaler is not None and label_encoders is not None:
            try:
                predictions_flat, appliance_data = predict_with_model(request.appliances, request.household_info)
                total_kwh = float(np.sum(predictions_flat))
                message = "Prediction completed using AI model with appliance-level analysis"
                
                # Create detailed breakdown
                breakdown_data = []
                for i, appliance_info in enumerate(appliance_data):
                    appliance_kwh = float(predictions_flat[i])
                    
                    breakdown_data.append({
                        "appliance": request.appliances[i].appliance,
                        "power_watts": appliance_info['Power_Watts'],
                        "hours_daily": request.appliances[i].hours,
                        "quantity": request.appliances[i].quantity,
                        "usage_days_monthly": request.appliances[i].usage_days_monthly,
                        "estimated_kwh": appliance_kwh,
                        "estimated_bill": 0,  # Will be calculated after total
                        "percentage": 0
                    })
                    
            except Exception as model_error:
                logger.warning(f"Model prediction failed: {model_error}")
                # Fallback to simple calculation
                total_kwh, breakdown_data = simple_energy_calculation(request.appliances)
                message = "Prediction completed using calculation (model unavailable)"
        else:
            # Use simple calculation
            total_kwh, breakdown_data = simple_energy_calculation(request.appliances)
            message = "Prediction completed using standard calculation"
        
        # Calculate final results
        total_kwh = round(total_kwh, 2)
        tariff_bracket = calculate_tariff_bracket(total_kwh)
        total_bill = calculate_estimated_bill(total_kwh)
        
        # Calculate appliance bills and percentages
        for item in breakdown_data:
            item['estimated_bill'] = (item['estimated_kwh'] / total_kwh) * total_bill if total_kwh > 0 else 0
            item['percentage'] = (item['estimated_kwh'] / total_kwh) * 100 if total_kwh > 0 else 0
            item['estimated_kwh'] = round(item['estimated_kwh'], 2)
            item['estimated_bill'] = round(item['estimated_bill'], 2)
            item['percentage'] = round(item['percentage'], 1)
        
        # Sort by consumption
        breakdown_data.sort(key=lambda x: x['estimated_kwh'], reverse=True)
        
        # Budget analysis
        budget_difference = total_bill - request.household_info.budget
        budget_status = "within_budget" if budget_difference <= 0 else "over_budget"
        
        # Generate recommendations
        recommendations = get_smart_recommendations(total_kwh, total_bill, breakdown_data, request.household_info.budget)
        
        # Convert to Pydantic models
        breakdown = [ApplianceResult(**item) for item in breakdown_data]
        
        return PredictionResponse(
            total_kwh=total_kwh,
            total_bill=total_bill,
            tariff_bracket=tariff_bracket,
            breakdown=breakdown,
            budget_status=budget_status,
            budget_difference=budget_difference,
            recommendations=recommendations,
            status="success",
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Error handlers
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
