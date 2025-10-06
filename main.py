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
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rwanda Energy Consumption Predictor API",
    description="API for predicting household energy consumption in Rwanda with detailed appliance analysis and unsupervised learning",
    version="3.0.0",
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

# Unsupervised Learning Models
class UnsupervisedModels:
    """Unsupervised learning models for household analysis"""
    
    def __init__(self):
        self.kmeans_model = None
        self.isolation_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.reports_storage = []
        
    def prepare_features(self, reports_data):
        """Prepare features from reports for unsupervised learning"""
        features = []
        
        for report in reports_data:
            # Extract numerical features from reports
            total_kwh = float(report.get('total_kwh', 0))
            total_bill = float(report.get('total_bill', 0))
            household_size = int(report.get('household_size', 1))
            
            # Calculate energy intensity (kWh per person)
            energy_intensity = total_kwh / household_size if household_size > 0 else total_kwh
            
            # Calculate bill intensity (RWF per kWh)
            bill_intensity = total_bill / total_kwh if total_kwh > 0 else 0
            
            feature_vector = [
                total_kwh,
                total_bill,
                household_size,
                energy_intensity,
                bill_intensity,
                len(report.get('appliances', [])),
                # Region encoding
                1 if report.get('region') == 'Kigali' else 0,
                1 if report.get('region') == 'Eastern' else 0,
                1 if report.get('region') == 'Western' else 0,
                1 if report.get('region') == 'Northern' else 0,
                1 if report.get('region') == 'Southern' else 0,
                # Income level encoding
                1 if report.get('income_level') == 'Low' else 0,
                1 if report.get('income_level') == 'Medium' else 0,
                1 if report.get('income_level') == 'High' else 0,
                # Urban vs Rural (Kigali is urban, others rural)
                1 if report.get('region') == 'Kigali' else 0,
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def fit_models(self, reports_data):
        """Fit both K-Means and Isolation Forest models with rule-based clustering"""
        if len(reports_data) < 3:
            return {"error": "Need at least 3 reports for clustering"}
        
        try:
            self.reports_storage = reports_data
            
            # Use rule-based clustering instead of K-Means
            clusters = []
            for report in reports_data:
                consumption = float(report.get('total_kwh', 0))
                
                # Rule-based clustering based on consumption ranges
                if consumption >= 20 and consumption <= 99:
                    clusters.append(0)  # Tier 1
                elif consumption >= 100 and consumption <= 250:
                    clusters.append(1)  # Tier 2
                elif consumption >= 300 and consumption <= 600:
                    clusters.append(2)  # Tier 3
                else:
                    # For consumption outside normal ranges, assign based on closest range
                    if consumption < 20:
                        clusters.append(0)  # Too low, assign to Tier 1
                    elif consumption > 600:
                        clusters.append(2)  # Too high, assign to Tier 3
                    else:
                        # For values between 251-299, assign to Tier 2
                        clusters.append(1)
            
            # Prepare features for anomaly detection only
            consumptions = [[float(r.get('total_kwh', 0))] for r in reports_data]
            X = np.array(consumptions)
            
            # Fit Isolation Forest with focus on high consumption only
            self.isolation_model = IsolationForest(
                contamination=0.05,  # Lower contamination for rare high-consumption anomalies
                random_state=42
            )
            anomalies = self.isolation_model.fit_predict(X)
            
            self.is_fitted = True
            
            return {
                "clusters": clusters,
                "anomalies": (anomalies == -1).tolist(),
                "cluster_centers": [[60], [175], [450]]  # Mock centers for compatibility
            }
            
        except Exception as e:
            return {"error": f"Model fitting failed: {str(e)}"}
    
    def predict_clusters(self, reports_data):
        """Predict clusters for new data using rule-based approach"""
        if not self.is_fitted:
            return {"error": "Models not fitted yet"}
        
        clusters = []
        for report in reports_data:
            consumption = float(report.get('total_kwh', 0))
            
            # Rule-based clustering based on consumption ranges
            if consumption >= 20 and consumption <= 99:
                clusters.append(0)  # Tier 1
            elif consumption >= 100 and consumption <= 250:
                clusters.append(1)  # Tier 2
            elif consumption >= 300 and consumption <= 600:
                clusters.append(2)  # Tier 3
            else:
                # For consumption outside normal ranges, assign based on closest range
                if consumption < 20:
                    clusters.append(0)  # Too low, assign to Tier 1
                elif consumption > 600:
                    clusters.append(2)  # Too high, assign to Tier 3
                else:
                    # For values between 251-299, assign to Tier 2
                    clusters.append(1)
        
        # For anomalies, use the fitted model
        consumptions = [[float(r.get('total_kwh', 0))] for r in reports_data]
        X = np.array(consumptions)
        anomalies = self.isolation_model.predict(X)
        
        return {
            "clusters": clusters,
            "anomalies": (anomalies == -1).tolist()
        }
    
    def get_cluster_descriptions(self):
        """Get descriptions for each cluster based on consumption ranges"""
        return {
            0: {
                "name": "Tier 1 - Small Families",
                "description": "Households with consumption 20-99 kWh/month - Basic needs, small families",
                "color": "green",
                "consumption_range": "20-99 kWh/month",
                "typical_profile": "1-3 persons, basic appliances"
            },
            1: {
                "name": "Tier 2 - Average Families", 
                "description": "Households with consumption 100-250 kWh/month - Average families with modern appliances",
                "color": "orange",
                "consumption_range": "100-250 kWh/month",
                "typical_profile": "3-5 persons, standard appliances"
            },
            2: {
                "name": "Tier 3 - Large Families",
                "description": "Households with consumption 300-600 kWh/month - Large families with multiple appliances",
                "color": "red", 
                "consumption_range": "300-600 kWh/month",
                "typical_profile": "5+ persons, multiple high-power appliances"
            }
        }
    
    def analyze_anomalies_simple(self, reports_data):
        """Simple rule-based anomaly detection for consumption over 600 kWh"""
        anomalies = []
        for i, report in enumerate(reports_data):
            consumption = float(report.get('total_kwh', 0))
            
            # Simple rule: consumption over 600 kWh is anomalous
            if consumption > 600:
                reasons = [f"Extreme consumption: {consumption:.1f} kWh/month (threshold: 600 kWh)"]
                
                # Add regional context
                region = report.get('region', 'Unknown')
                region_avg = self.calculate_region_average(region, reports_data)
                if region_avg:
                    reasons.append(f"Consumption {consumption/region_avg:.1f}x higher than {region} average ({region_avg:.1f} kWh)")
                
                # Add household size context
                household_size = int(report.get('household_size', 1))
                consumption_per_person = consumption / household_size if household_size > 0 else consumption
                if consumption_per_person > 200:
                    reasons.append(f"Very high per capita consumption: {consumption_per_person:.1f} kWh/person")
                
                anomalies.append({
                    "report": report,
                    "anomaly_score": 0.7,
                    "reasons": reasons,
                    "consumption_per_person": consumption_per_person,
                    "region_comparison": region_avg
                })
        
        return anomalies
    
    def calculate_region_average(self, region, reports_data):
        """Calculate average consumption for a region"""
        region_consumptions = [float(r.get('total_kwh', 0)) for r in reports_data if r.get('region') == region]
        return np.mean(region_consumptions) if region_consumptions else None
    
    def get_all_reports(self):
        """Get all stored reports"""
        return self.reports_storage

# Initialize unsupervised models
unsupervised_models = UnsupervisedModels()

def initialize_models():
    """Initialize all models and preprocessing objects"""
    global model, scaler, label_encoders, feature_names, X_train_columns
    
    try:
        logger.info("Loading model and preprocessing objects...")
        
        # Load preprocessing objects
        scaler = joblib.load('energy_scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        X_train_columns = joblib.load('X_train_columns.pkl')
        
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
        logger.error(f" Error loading models: {e}")
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
    report_id: str

# Unsupervised Learning Pydantic Models
class ClusterAnalysisRequest(BaseModel):
    reports: List[Dict[str, Any]]

class ClusterResult(BaseModel):
    cluster_id: int
    cluster_name: str
    description: str
    color: str
    households: List[Dict[str, Any]]
    avg_consumption: float
    avg_bill: float
    size: int
    consumption_range: str
    typical_profile: str
    min_consumption: float
    max_consumption: float
    common_regions: List[str]

class AnomalyResult(BaseModel):
    report: Dict[str, Any]
    anomaly_score: float
    reasons: List[str]
    consumption_per_person: float
    region_comparison: Optional[float]

class UnsupervisedResponse(BaseModel):
    status: str
    message: str
    clusters: List[ClusterResult]
    anomalies: List[AnomalyResult]
    total_households: int
    model_metrics: Dict[str, Any]

class ReportStorage(BaseModel):
    reports: List[Dict[str, Any]]

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
        message=" BUDGET ANALYSIS"
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
        message=" TARIFF BRACKET INSIGHTS"
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
            message=" Consider using high-power appliances during off-peak hours"
        ))
    
    high_consumption = [item for item in breakdown if item['estimated_kwh'] > 10]
    if high_consumption:
        recommendations.append(RecommendationItem(
            category="optimization",
            message=" Focus on reducing usage for high-consumption appliances first"
        ))
    
    return recommendations

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Rwanda Energy Consumption Predictor API",
        "version": "3.0.0",
        "status": "running",
        "description": "Advanced energy prediction with appliance-level analysis and unsupervised learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "unsupervised": "/api/unsupervised/*",
            "reports": "/api/reports/*"
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
        "unsupervised_ready": unsupervised_models.is_fitted,
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
        
        # Generate report ID
        report_id = str(uuid.uuid4())[:8]
        
        # Store report for unsupervised learning
        report_data = {
            "id": report_id,
            "timestamp": datetime.now().isoformat(),
            "total_kwh": total_kwh,
            "total_bill": total_bill,
            "tariff_bracket": tariff_bracket,
            "household_size": request.household_info.household_size,
            "region": request.household_info.region,
            "income_level": request.household_info.income_level,
            "budget": request.household_info.budget,
            "appliances": [app.dict() for app in request.appliances],
            "breakdown": [item.dict() for item in breakdown]
        }
        
        # Add to unsupervised models storage
        unsupervised_models.reports_storage.append(report_data)
        
        return PredictionResponse(
            total_kwh=total_kwh,
            total_bill=total_bill,
            tariff_bracket=tariff_bracket,
            breakdown=breakdown,
            budget_status=budget_status,
            budget_difference=budget_difference,
            recommendations=recommendations,
            status="success",
            message=message,
            report_id=report_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Unsupervised Learning Endpoints
@app.post("/api/unsupervised/cluster", response_model=UnsupervisedResponse)
async def cluster_households(request: ClusterAnalysisRequest):
    """
    Perform rule-based clustering on household reports
    
    Groups households into 3 clusters based on consumption ranges:
    - Tier 1: 20-99 kWh/month (Small Families)
    - Tier 2: 100-250 kWh/month (Average Families)  
    - Tier 3: 300-600 kWh/month (Large Families)
    """
    try:
        if len(request.reports) < 3:
            raise HTTPException(
                status_code=400, 
                detail="At least 3 reports are required for clustering"
            )
        
        # Fit models
        fit_result = unsupervised_models.fit_models(request.reports)
        
        if "error" in fit_result:
            raise HTTPException(status_code=400, detail=fit_result["error"])
        
        # Organize results into clusters
        clusters = []
        cluster_descriptions = unsupervised_models.get_cluster_descriptions()
        
        # Create cluster results - use the original reports with all their data
        for cluster_id in range(3):
            cluster_reports = [
                report for i, report in enumerate(request.reports) 
                if fit_result["clusters"][i] == cluster_id
            ]
            
            if cluster_reports:
                consumptions = [float(r.get('total_kwh', 0)) for r in cluster_reports]
                bills = [float(r.get('total_bill', 0)) for r in cluster_reports]
                sizes = [int(r.get('household_size', 1)) for r in cluster_reports]
                regions = [r.get('region', 'Unknown') for r in cluster_reports]
                income_levels = [r.get('income_level', 'Unknown') for r in cluster_reports]
                
                avg_consumption = np.mean(consumptions)
                avg_bill = np.mean(bills)
                min_consumption = np.min(consumptions)
                max_consumption = np.max(consumptions)
                
                # Get most common regions and income levels
                common_regions = pd.Series(regions).value_counts().index.tolist()[:3]
                common_income_levels = pd.Series(income_levels).value_counts().index.tolist()[:3]
                
            else:
                avg_consumption = 0
                avg_bill = 0
                min_consumption = 0
                max_consumption = 0
                common_regions = []
                common_income_levels = []
            
            cluster_info = cluster_descriptions[cluster_id]
            
            # Create enhanced cluster description based on actual data
            enhanced_description = f"{cluster_info['description']}"
            if cluster_reports:
                avg_household_size = np.mean(sizes)
                enhanced_description += f" - Average household size: {avg_household_size:.1f} persons"
            
            clusters.append(ClusterResult(
                cluster_id=cluster_id,
                cluster_name=cluster_info["name"],
                description=enhanced_description,
                color=cluster_info["color"],
                households=cluster_reports,  # This now contains all original report data
                avg_consumption=round(avg_consumption, 2),
                avg_bill=round(avg_bill, 2),
                size=len(cluster_reports),
                consumption_range=cluster_info["consumption_range"],
                typical_profile=cluster_info["typical_profile"],
                min_consumption=round(min_consumption, 2),
                max_consumption=round(max_consumption, 2),
                common_regions=common_regions
            ))
        
        # Identify anomalies using simple rule-based approach (600+ kWh only)
        anomalies_data = unsupervised_models.analyze_anomalies_simple(request.reports)
        
        # Convert to Pydantic models
        anomalies = []
        for anomaly_data in anomalies_data:
            anomalies.append(AnomalyResult(
                report=anomaly_data["report"],
                anomaly_score=anomaly_data["anomaly_score"],
                reasons=anomaly_data["reasons"],
                consumption_per_person=round(anomaly_data["consumption_per_person"], 2),
                region_comparison=round(anomaly_data["region_comparison"], 2) if anomaly_data["region_comparison"] else None
            ))
        
        return UnsupervisedResponse(
            status="success",
            message=f"Clustered {len(request.reports)} households into 3 consumption tiers using rule-based approach",
            clusters=clusters,
            anomalies=anomalies,
            total_households=len(request.reports),
            model_metrics={
                "clusters_identified": 3,
                "anomalies_detected": len(anomalies),
                "anomaly_rate": f"{(len(anomalies) / len(request.reports)) * 100:.1f}%",
                "average_consumption": f"{np.mean([float(r.get('total_kwh', 0)) for r in request.reports]):.1f} kWh",
                "average_bill": f"{np.mean([float(r.get('total_bill', 0)) for r in request.reports]):.0f} RWF",
                "high_consumption_anomalies": len([a for a in anomalies if a.report.get('total_kwh', 0) > 600]),
                "clustering_method": "Rule-based (Consumption Ranges)"
            }
        )
        
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.get("/api/unsupervised/stats")
async def get_unsupervised_stats():
    """Get statistics about the unsupervised models"""
    return {
        "status": "success",
        "models_loaded": unsupervised_models.is_fitted,
        "kmeans_ready": unsupervised_models.kmeans_model is not None,
        "isolation_forest_ready": unsupervised_models.isolation_model is not None,
        "total_reports_stored": len(unsupervised_models.reports_storage),
        "last_analysis": "Never" if not unsupervised_models.is_fitted else "Completed"
    }

@app.get("/api/unsupervised/reports")
async def get_all_reports():
    """Get all stored reports for analysis"""
    return {
        "status": "success",
        "total_reports": len(unsupervised_models.reports_storage),
        "reports": unsupervised_models.get_all_reports()
    }

@app.post("/api/unsupervised/analyze-stored")
async def analyze_stored_reports():
    """Analyze all stored reports using unsupervised learning"""
    try:
        reports = unsupervised_models.get_all_reports()
        if len(reports) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 stored reports for analysis")
        
        # Create request for clustering
        request = ClusterAnalysisRequest(reports=reports)
        return await cluster_households(request)
        
    except Exception as e:
        logger.error(f"Stored analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Reports Management Endpoints
@app.get("/api/reports")
async def get_all_prediction_reports():
    """Get all prediction reports generated by the system"""
    return {
        "status": "success",
        "total_reports": len(unsupervised_models.reports_storage),
        "reports": unsupervised_models.reports_storage
    }

@app.get("/api/reports/{report_id}")
async def get_report_by_id(report_id: str):
    """Get a specific report by ID"""
    for report in unsupervised_models.reports_storage:
        if report.get('id') == report_id:
            return {
                "status": "success",
                "report": report
            }
    
    raise HTTPException(status_code=404, detail="Report not found")

@app.delete("/api/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a specific report by ID"""
    global unsupervised_models
    
    for i, report in enumerate(unsupervised_models.reports_storage):
        if report.get('id') == report_id:
            deleted_report = unsupervised_models.reports_storage.pop(i)
            return {
                "status": "success",
                "message": f"Report {report_id} deleted successfully",
                "deleted_report": deleted_report
            }
    
    raise HTTPException(status_code=404, detail="Report not found")

@app.delete("/api/reports")
async def clear_all_reports():
    """Clear all stored reports"""
    global unsupervised_models
    
    deleted_count = len(unsupervised_models.reports_storage)
    unsupervised_models.reports_storage = []
    unsupervised_models.is_fitted = False
    unsupervised_models.kmeans_model = None
    unsupervised_models.isolation_model = None
    
    return {
        "status": "success",
        "message": f"All {deleted_count} reports cleared successfully",
        "reports_remaining": 0
    }

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
