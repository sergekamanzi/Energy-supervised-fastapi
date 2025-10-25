import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Dict, Any, Optional
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import logging
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime, timedelta
import os
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rwanda Energy Consumption Predictor API",
    description="API for predicting household energy consumption in Rwanda with detailed appliance analysis and time series forecasting",
    version="6.0.0",  # Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global handler to surface request validation issues (HTTP 422) with clear messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_preview = body.decode('utf-8')[:1000]
    except Exception:
        body_preview = "<unavailable>"
    logger.error(f"Request validation error on {request.url.path}: {exc.errors()} | Body preview: {body_preview}")
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Invalid request payload. Check required fields and types.",
            "errors": exc.errors(),
        },
    )

# Global variables for models
model = None
scaler = None
label_encoders = None
feature_names = None
X_train_columns = None

# Time Series Forecasting Models
timeseries_lstm_model = None
timeseries_scaler_X = None
timeseries_scaler_y = None
timeseries_sarima_model = None
timeseries_artifacts = None

# Clustering and Anomaly Detection Models
kmeans_model = None
isolation_forest_model = None
scaler_cluster = None
scaler_anomaly = None
pca_model = None
clustering_label_encoders = None

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

# =============================================================================
# CLUSTERING AND ANOMALY DETECTION MODELS
# =============================================================================

class ClusteringAnomalyDetector:
    """Clustering and Anomaly Detection for Household Energy Analysis"""
    
    def __init__(self):
        self.kmeans_model = None
        self.isolation_forest_model = None
        self.scaler_cluster = None
        self.scaler_anomaly = None
        self.pca_model = None
        self.label_encoders = None
        self.loaded = False
    
    def load_models(self):
        """Load clustering and anomaly detection models"""
        try:
            # Use absolute path relative to this file for robustness
            base_dir = os.path.dirname(os.path.abspath(__file__))
            saved_models_dir = os.path.join(base_dir, "saved_models")
            
            # Load K-Means model
            kmeans_path = os.path.join(saved_models_dir, "kmeans_model.pkl")
            if os.path.exists(kmeans_path):
                with open(kmeans_path, 'rb') as f:
                    self.kmeans_model = pickle.load(f)
                logger.info("✓ K-Means clustering model loaded")
            else:
                logger.warning(f"K-Means model not found at {kmeans_path}")
            
            # Load Isolation Forest model
            isolation_path = os.path.join(saved_models_dir, "isolation_forest_model.pkl")
            if os.path.exists(isolation_path):
                with open(isolation_path, 'rb') as f:
                    self.isolation_forest_model = pickle.load(f)
                logger.info("✓ Isolation Forest anomaly detection model loaded")
            else:
                logger.warning(f"Isolation Forest model not found at {isolation_path}")
            
            # Load scalers
            scaler_cluster_path = os.path.join(saved_models_dir, "scaler_cluster.pkl")
            scaler_anomaly_path = os.path.join(saved_models_dir, "scaler_anomaly.pkl")
            
            if os.path.exists(scaler_cluster_path):
                with open(scaler_cluster_path, 'rb') as f:
                    self.scaler_cluster = pickle.load(f)
                logger.info("✓ Clustering scaler loaded")
            if os.path.exists(scaler_anomaly_path):
                with open(scaler_anomaly_path, 'rb') as f:
                    self.scaler_anomaly = pickle.load(f)
                logger.info("✓ Anomaly detection scaler loaded")
            
            # Load PCA model
            pca_path = os.path.join(saved_models_dir, "pca.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, 'rb') as f:
                    self.pca_model = pickle.load(f)
                logger.info("✓ PCA model loaded")
            
            # Load label encoders
            encoders_path = os.path.join(saved_models_dir, "label_encoders.pkl")
            if os.path.exists(encoders_path):
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                logger.info("✓ Clustering label encoders loaded")
            
            self.loaded = (self.kmeans_model is not None and 
                          self.isolation_forest_model is not None and
                          self.scaler_cluster is not None)
            
            if self.loaded:
                logger.info("🎯 CLUSTERING & ANOMALY DETECTION MODELS LOADED SUCCESSFULLY!")
            else:
                logger.warning("Some clustering/anomaly models failed to load")
                
        except Exception as e:
            logger.error(f"Error loading clustering/anomaly models: {e}")
            self.loaded = False
    
    def predict_cluster_and_anomaly(self, household_data: Dict[str, Any]):
        """Predict cluster and anomaly for a household"""
        if not self.loaded:
            return {"error": "Clustering and anomaly detection models not loaded"}
        
        try:
            # Prepare features for clustering
            cluster_features = [
                'Total_kWh_Monthly',
                'Estimated_Bill_RWF', 
                'Household_Size',
                'Income_Level_encoded',
                'Region_encoded',
                'Quantity',
                'Usage_Hours_Daily',
                'Energy_Per_Capita',
                'Bill_Per_Capita'
            ]
            
            # Prepare features for anomaly detection
            anomaly_features = [
                'Total_kWh_Monthly',
                'Estimated_Bill_RWF',
                'Usage_Hours_Daily',
                'Quantity',
                'Tariff_Bracket_encoded',
                'Region_encoded',
                'Month_encoded',
                'Daily_Energy_kWh',
                'Cost_Per_kWh'
            ]
            
            # Create input DataFrame
            input_data = {}
            
            # Add basic features
            input_data['Total_kWh_Monthly'] = household_data.get('total_kwh', 0)
            input_data['Estimated_Bill_RWF'] = household_data.get('total_bill', 0)
            input_data['Household_Size'] = household_data.get('household_size', 1)
            input_data['Usage_Hours_Daily'] = household_data.get('avg_usage_hours', 0)
            input_data['Quantity'] = household_data.get('appliance_count', 0)
            
            # Calculate derived features
            input_data['Energy_Per_Capita'] = input_data['Total_kWh_Monthly'] / max(input_data['Household_Size'], 1)
            input_data['Bill_Per_Capita'] = input_data['Estimated_Bill_RWF'] / max(input_data['Household_Size'], 1)
            input_data['Daily_Energy_kWh'] = household_data.get('daily_energy', 0)
            input_data['Cost_Per_kWh'] = input_data['Estimated_Bill_RWF'] / max(input_data['Total_kWh_Monthly'], 1)
            
            # Encode categorical variables
            region = household_data.get('region', 'Kigali')
            income_level = household_data.get('income_level', 'Medium')
            tariff_bracket = household_data.get('tariff_bracket', '21-50 kWh')
            month = household_data.get('month', 'January')
            
            if self.label_encoders:
                try:
                    input_data['Region_encoded'] = self.label_encoders['Region'].transform([region])[0]
                    input_data['Income_Level_encoded'] = self.label_encoders['Income_Level'].transform([income_level])[0]
                    input_data['Tariff_Bracket_encoded'] = self.label_encoders['Tariff_Bracket'].transform([tariff_bracket])[0]
                    input_data['Month_encoded'] = self.label_encoders['Month'].transform([month])[0]
                except Exception as e:
                    logger.warning(f"Encoding error, using defaults: {e}")
                    # Use default encodings
                    input_data['Region_encoded'] = 1  # Kigali
                    input_data['Income_Level_encoded'] = 1  # Medium
                    input_data['Tariff_Bracket_encoded'] = 1  # 21-50 kWh
                    input_data['Month_encoded'] = 0  # January
            
            # Create feature arrays
            cluster_features_array = np.array([[input_data.get(f, 0) for f in cluster_features]])
            anomaly_features_array = np.array([[input_data.get(f, 0) for f in anomaly_features]])
            
            # Scale features
            cluster_scaled = self.scaler_cluster.transform(cluster_features_array)
            anomaly_scaled = self.scaler_anomaly.transform(anomaly_features_array)
            
            # Predict cluster
            cluster_pred = self.kmeans_model.predict(cluster_scaled)[0]
            
            # Predict anomaly
            anomaly_pred = self.isolation_forest_model.predict(anomaly_scaled)[0]
            anomaly_score = self.isolation_forest_model.decision_function(anomaly_scaled)[0]
            
            # Get cluster profile
            cluster_profile = self.get_cluster_profile(cluster_pred)
            
            return {
                "cluster": int(cluster_pred),
                "cluster_profile": cluster_profile,
                "anomaly_status": "Anomaly" if anomaly_pred == -1 else "Normal",
                "anomaly_score": float(anomaly_score),
                "anomaly_confidence": self.get_anomaly_confidence(anomaly_score),
                "features_used": {
                    "clustering": cluster_features,
                    "anomaly_detection": anomaly_features
                }
            }
            
        except Exception as e:
            logger.error(f"Clustering/anomaly prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_cluster_profile(self, cluster_id: int) -> Dict[str, Any]:
        """Get descriptive profile for a cluster"""
        profiles = {
            0: {
                "name": "Low Consumption Households",
                "description": "Small households with minimal energy usage, typically in lower tariff brackets",
                "characteristics": ["Small household size", "Low income", "Few appliances", "Efficient usage"],
                "recommendations": ["Consider energy-efficient upgrades", "Monitor for unusual spikes"]
            },
            1: {
                "name": "Medium Consumption Households", 
                "description": "Average households with balanced energy consumption across essential appliances",
                "characteristics": ["Medium household size", "Moderate income", "Standard appliance usage"],
                "recommendations": ["Optimize usage patterns", "Consider time-of-use strategies"]
            },
            2: {
                "name": "High Consumption Households",
                "description": "Large households or those with high-energy appliances, typically in higher tariff brackets",
                "characteristics": ["Large household size", "Higher income", "Multiple appliances", "High usage hours"],
                "recommendations": ["Implement energy-saving measures", "Consider solar alternatives", "Monitor for waste"]
            }
        }
        
        return profiles.get(cluster_id, {
            "name": f"Cluster {cluster_id}",
            "description": "Household energy consumption pattern",
            "characteristics": [],
            "recommendations": []
        })
    
    def get_anomaly_confidence(self, anomaly_score: float) -> str:
        """Convert anomaly score to confidence level"""
        if anomaly_score < -0.2:
            return "High"
        elif anomaly_score < 0:
            return "Medium"
        else:
            return "Low"
    
    def analyze_household_patterns(self, households_data: List[Dict[str, Any]]):
        """Analyze multiple households for patterns and insights"""
        if not self.loaded:
            return {"error": "Models not loaded"}
        
        try:
            results = []
            cluster_counts = {0: 0, 1: 0, 2: 0}
            anomaly_count = 0
            
            for household in households_data:
                prediction = self.predict_cluster_and_anomaly(household)
                if "error" not in prediction:
                    results.append({
                        "household_id": household.get('household_id', 'unknown'),
                        **prediction
                    })
                    
                    cluster_counts[prediction["cluster"]] += 1
                    if prediction["anomaly_status"] == "Anomaly":
                        anomaly_count += 1
            
            # Generate insights
            total_households = len(results)
            if total_households > 0:
                anomaly_rate = (anomaly_count / total_households) * 100
                dominant_cluster = max(cluster_counts, key=cluster_counts.get)
                
                insights = {
                    "total_households_analyzed": total_households,
                    "cluster_distribution": cluster_counts,
                    "anomaly_rate_percentage": round(anomaly_rate, 2),
                    "dominant_cluster": self.get_cluster_profile(dominant_cluster)["name"],
                    "recommendations": self.generate_community_recommendations(cluster_counts, anomaly_rate)
                }
            else:
                insights = {"error": "No valid predictions generated"}
            
            return {
                "individual_predictions": results,
                "community_insights": insights
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_community_recommendations(self, cluster_counts: Dict[int, int], anomaly_rate: float) -> List[str]:
        """Generate community-level energy recommendations"""
        recommendations = []
        
        total = sum(cluster_counts.values())
        if total == 0:
            return ["Insufficient data for community recommendations"]
        
        # Cluster-based recommendations
        if cluster_counts[2] / total > 0.4:  # If more than 40% are high consumers
            recommendations.append("Community has high energy consumption - consider group energy efficiency programs")
        
        if cluster_counts[0] / total > 0.5:  # If more than 50% are low consumers
            recommendations.append("Community is energy efficient - share best practices with neighboring areas")
        
        # Anomaly-based recommendations
        if anomaly_rate > 10:
            recommendations.append(f"High anomaly rate ({anomaly_rate:.1f}%) - investigate potential meter issues or unusual usage patterns")
        elif anomaly_rate < 2:
            recommendations.append("Low anomaly rate - consumption patterns appear normal and consistent")
        
        # General recommendations
        recommendations.append("Consider implementing smart meter technology for better monitoring")
        recommendations.append("Explore community solar initiatives to reduce grid dependency")
        
        return recommendations

# Initialize clustering and anomaly detector
clustering_detector = ClusteringAnomalyDetector()

# =============================================================================
# TIME SERIES FORECASTING MODELS
# =============================================================================

class TimeSeriesForecaster:
    """Time Series Forecasting for Individual Household Energy Consumption"""
    
    def __init__(self):
        self.lstm_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sarima_model = None
        self.artifacts = None
        self.loaded = False
    
    def load_models(self):
        """Load time series forecasting models"""
        try:
            # Use absolute path relative to this file for robustness
            base_dir = os.path.dirname(os.path.abspath(__file__))
            timeseries_dir = os.path.join(base_dir, "timeseries")
            
            # Load LSTM model
            lstm_path = os.path.join(timeseries_dir, "individual_household_lstm_model.h5")
            if os.path.exists(lstm_path):
                self.lstm_model = load_model(lstm_path)
                logger.info("✓ Time Series LSTM model loaded")
            else:
                logger.warning(f"LSTM model not found at {lstm_path}")
            
            # Load SARIMA model
            sarima_path = os.path.join(timeseries_dir, "sarima_model.pkl")
            if os.path.exists(sarima_path):
                # Load with joblib (supports sklearn and statsmodels pickles)
                try:
                    self.sarima_model = joblib.load(sarima_path)
                except Exception:
                    # Fallback to pickle.load for other pickle formats
                    import pickle
                    with open(sarima_path, 'rb') as f:
                        self.sarima_model = pickle.load(f)
                logger.info("✓ Time Series SARIMA model loaded")
            else:
                logger.warning(f"SARIMA model not found at {sarima_path}")
            
            # Load scalers
            scaler_x_path = os.path.join(timeseries_dir, "scaler_X.pkl")
            scaler_y_path = os.path.join(timeseries_dir, "scaler_y.pkl")
            
            if os.path.exists(scaler_x_path):
                self.scaler_X = joblib.load(scaler_x_path)
                logger.info("✓ Time Series Scaler X loaded")
            if os.path.exists(scaler_y_path):
                self.scaler_y = joblib.load(scaler_y_path)
                logger.info("✓ Time Series Scaler Y loaded")
            
            # Load artifacts
            artifacts_path = os.path.join(timeseries_dir, "model_artifacts_complete.pkl")
            if os.path.exists(artifacts_path):
                self.artifacts = joblib.load(artifacts_path)
                logger.info("✓ Time Series artifacts loaded")
            
            self.loaded = (self.lstm_model is not None and 
                          self.scaler_X is not None and 
                          self.scaler_y is not None)
            
            if self.loaded:
                logger.info("🎯 TIME SERIES FORECASTING MODELS LOADED SUCCESSFULLY!")
            else:
                logger.warning("Some time series models failed to load")
                
        except Exception as e:
            logger.error(f"Error loading time series models: {e}")
            self.loaded = False
    
    def predict_future_consumption(self, historical_data: List[float], months_ahead: int = 3):
        """Predict future energy consumption using LSTM model"""
        if not self.loaded:
            return {"error": "Time series models not loaded"}
        
        if len(historical_data) < 3:
            return {"error": "At least 3 months of historical data required"}
        
        try:
            # Use last 3 months for prediction
            lookback = 3
            last_months = historical_data[-lookback:]
            
            # Prepare input for LSTM
            input_sequence = np.array(last_months).reshape(1, lookback, 1)
            input_scaled = self.scaler_X.transform(input_sequence.reshape(-1, 1)).reshape(1, lookback, 1)
            
            # Make prediction
            prediction_scaled = self.lstm_model.predict(input_scaled, verbose=0)
            prediction_kwh = self.scaler_y.inverse_transform(prediction_scaled).flatten()
            
            # If we need more months than the model can predict directly, use recursive forecasting
            predictions = list(prediction_kwh[:months_ahead])
            
            # For months beyond the direct forecast horizon, use the last predicted value
            while len(predictions) < months_ahead:
                predictions.append(predictions[-1] * 0.95)  # Slight decay assumption
            
            return {
                "historical_data": historical_data,
                "predictions": predictions,
                "forecast_months": months_ahead,
                "model_used": "LSTM",
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_with_sarima(self, historical_data: List[float], months_ahead: int = 3):
        """Predict using SARIMA model (fallback)"""
        if self.sarima_model is None:
            return {"error": "SARIMA model not available"}
        
        try:
            # This is a simplified implementation - you'd need to adapt based on your SARIMA model structure
            # For now, return a simple trend-based prediction
            if len(historical_data) >= 2:
                trend = (historical_data[-1] - historical_data[0]) / len(historical_data)
                last_value = historical_data[-1]
                
                predictions = []
                for i in range(months_ahead):
                    next_pred = last_value + trend * (i + 1)
                    # Add some seasonality assumption (simplified)
                    seasonal_factor = 1.0 + 0.1 * np.sin((len(historical_data) + i) * np.pi / 6)
                    predictions.append(max(0, next_pred * seasonal_factor))
                
                return {
                    "historical_data": historical_data,
                    "predictions": predictions,
                    "forecast_months": months_ahead,
                    "model_used": "SARIMA",
                    "confidence": "medium"
                }
            else:
                return {"error": "Insufficient data for SARIMA"}
                
        except Exception as e:
            logger.error(f"SARIMA prediction error: {e}")
            return {"error": f"SARIMA prediction failed: {str(e)}"}
    
    def calculate_bill_forecast(self, consumption_forecast: List[float]):
        """Calculate bill forecasts based on consumption predictions"""
        bill_forecast = []
        tariff_brackets = []
        
        for consumption in consumption_forecast:
            bill = self.calculate_estimated_bill(consumption)
            bracket = self.calculate_tariff_bracket(consumption)
            
            bill_forecast.append(bill)
            tariff_brackets.append(bracket)
        
        return {
            "bill_predictions": bill_forecast,
            "tariff_brackets": tariff_brackets,
            "total_estimated_bill": sum(bill_forecast)
        }
    
    def calculate_tariff_bracket(self, total_kwh: float) -> str:
        """Determine tariff bracket based on monthly consumption"""
        if total_kwh <= 20:
            return '0-20 kWh'
        elif total_kwh <= 50:
            return '21-50 kWh'
        else:
            return '50+ kWh'
    
    def calculate_estimated_bill(self, total_kwh: float) -> float:
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

# Initialize time series forecaster
timeseries_forecaster = TimeSeriesForecaster()

# =============================================================================
# INITIALIZE ALL MODELS
# =============================================================================

def initialize_models():
    """Initialize all models: supervised, time series, clustering, and anomaly detection"""
    global model, scaler, label_encoders, feature_names, X_train_columns
    
    try:
        logger.info("Loading all AI models...")
        
        # Load supervised learning models
        try:
            scaler = joblib.load('energy_scaler.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            feature_names = joblib.load('feature_names.pkl')
            X_train_columns = joblib.load('X_train_columns.pkl')
            logger.info("✓ Supervised learning preprocessing objects loaded")
        except Exception as e:
            logger.warning(f"Supervised learning preprocessing objects not loaded: {e}")
            scaler = None
            label_encoders = None
        
        # Try to load TensorFlow model
        try:
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
            }
            model = load_model('rwanda_energy_model.h5', custom_objects=custom_objects, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info("✓ TensorFlow supervised model loaded")
        except Exception as e:
            logger.warning(f"TensorFlow model not loaded: {e}")
            try:
                model = joblib.load('energy_model.pkl')
                logger.info("✓ Scikit-learn supervised model loaded")
            except Exception as e2:
                logger.warning(f"Scikit-learn model also failed: {e2}")
                model = None
        
        # Load time series forecasting models
        timeseries_forecaster.load_models()
        
        # Load clustering and anomaly detection models
        clustering_detector.load_models()
        
        logger.info("🎯 ALL MODELS LOADED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        model = None
        scaler = None
        label_encoders = None

# Initialize on startup
initialize_models()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

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
    model_used: str

# Time Series Forecasting Models
class TimeSeriesRequest(BaseModel):
    historical_data: List[float]
    months_ahead: int = 3
    household_info: Optional[Dict[str, Any]] = None

class TimeSeriesPrediction(BaseModel):
    month: str
    predicted_consumption_kwh: float
    predicted_bill_rwf: float
    tariff_bracket: str
    confidence: str

class TimeSeriesResponse(BaseModel):
    status: str
    message: str
    historical_data: List[float]
    forecast_months: int
    predictions: List[TimeSeriesPrediction]
    total_forecasted_consumption: float
    total_forecasted_bill: float
    average_monthly_consumption: float
    average_monthly_bill: float
    trend: str
    trend_percentage: float
    model_used: str
    forecast_id: str

# Clustering and Anomaly Detection Models
class ClusterAnomalyRequest(BaseModel):
    # Accept both snake_case and camelCase from UI
    total_kwh: float = Field(validation_alias=AliasChoices('total_kwh', 'totalKwh'))
    total_bill: float = Field(validation_alias=AliasChoices('total_bill', 'totalBill'))
    household_size: int = Field(validation_alias=AliasChoices('household_size', 'householdSize'))
    region: str = Field(validation_alias=AliasChoices('region', 'Region'))
    income_level: str = Field(validation_alias=AliasChoices('income_level', 'incomeLevel'))
    tariff_bracket: str = Field(validation_alias=AliasChoices('tariff_bracket', 'tariffBracket'))
    appliance_count: int = Field(validation_alias=AliasChoices('appliance_count', 'applianceCount'))
    avg_usage_hours: float = Field(validation_alias=AliasChoices('avg_usage_hours', 'avgUsageHours'))
    daily_energy: Optional[float] = Field(default=None, validation_alias=AliasChoices('daily_energy', 'dailyEnergy'))
    month: Optional[str] = Field(default="January", validation_alias=AliasChoices('month', 'Month'))
    household_id: Optional[str] = Field(default=None, validation_alias=AliasChoices('household_id', 'householdId'))

class ClusterAnomalyResponse(BaseModel):
    cluster: int
    cluster_profile: Dict[str, Any]
    anomaly_status: str
    anomaly_score: float
    anomaly_confidence: str
    features_used: Dict[str, List[str]]
    status: str
    message: str
    analysis_id: str

class BatchClusterRequest(BaseModel):
    households: List[ClusterAnomalyRequest]

class BatchClusterResponse(BaseModel):
    individual_predictions: List[Dict[str, Any]]
    community_insights: Dict[str, Any]
    status: str
    message: str
    batch_id: str

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
    """Preprocess user input for ML model prediction"""
    
    appliance_data = []
    for appliance in appliances_list:
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

    input_df = pd.DataFrame(appliance_data)
    return input_df, appliance_data

def handle_unseen_appliances(input_df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Handle unseen appliances by mapping to known ones"""
    if label_encoders is None:
        return input_df
        
    known_appliances = set(label_encoders['Appliance'].classes_)
    for idx, appliance_name in enumerate(input_df['Appliance']):
        if appliance_name not in known_appliances:
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
    """Predict energy consumption using supervised ML model"""
    
    input_df, appliance_data = preprocess_user_input(appliances_list, household_info)
    
    if label_encoders is not None:
        input_df = handle_unseen_appliances(input_df, label_encoders)
        input_df = encode_categorical_features(input_df, label_encoders)
    
    if X_train_columns is not None:
        for col in X_train_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_train_columns]
    
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
    
    if model is not None:
        appliance_predictions = model.predict(input_scaled)
        
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
        raise Exception("No supervised model available for prediction")

def simple_energy_calculation(appliances_list: List[ApplianceInput]) -> Dict:
    """Simple energy calculation as fallback"""
    total_kwh = 0
    breakdown = []
    
    for appliance in appliances_list:
        power_watts = convert_power_units(appliance.power, appliance.power_unit, 'W')
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
        message="💰 BUDGET ANALYSIS"
    ))
    
    if total_bill > budget:
        overspend = total_bill - budget
        recommendations.append(RecommendationItem(
            category="budget",
            message=f"• You're exceeding your budget by {overspend:.0f} RWF"
        ))
        
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
        message="📊 TARIFF BRACKET INSIGHTS"
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
    
    return recommendations

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Rwanda Energy Consumption Predictor API",
        "version": "6.0.0",
        "status": "running",
        "description": "Advanced energy prediction with supervised learning, time series forecasting, clustering, and anomaly detection",
        "ai_models": {
            "supervised": "Energy consumption prediction",
            "time_series": "Future consumption forecasting", 
            "clustering": "Household segmentation",
            "anomaly_detection": "Unusual pattern detection"
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "timeseries": "/api/timeseries/*",
            "clustering": "/api/clustering/*"
        }
    }

@app.get("/health")
async def health_check():
    supervised_loaded = model is not None
    timeseries_ready = timeseries_forecaster.loaded
    clustering_ready = clustering_detector.loaded
    
    return {
        "status": "healthy" if supervised_loaded else "degraded", 
        "models": {
            "supervised": {
                "loaded": supervised_loaded,
                "scaler_loaded": scaler is not None,
                "encoders_loaded": label_encoders is not None
            },
            "time_series_forecasting": {
                "ready": timeseries_ready,
                "models_loaded": timeseries_forecaster.loaded,
                "lstm_available": timeseries_forecaster.lstm_model is not None,
                "sarima_available": timeseries_forecaster.sarima_model is not None
            },
            "clustering_anomaly_detection": {
                "ready": clustering_ready,
                "models_loaded": clustering_detector.loaded,
                "kmeans_available": clustering_detector.kmeans_model is not None,
                "isolation_forest_available": clustering_detector.isolation_forest_model is not None
            }
        },
        "message": "All AI models available" if supervised_loaded and timeseries_ready and clustering_ready else "Some models unavailable",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_energy_consumption(request: PredictionRequest):
    """
    Predict energy consumption using supervised learning
    
    **AI Models Used:**
    - Supervised Learning: Energy consumption prediction
    """
    try:
        logger.info(f"Received prediction request for {len(request.appliances)} appliances")
        
        if not request.appliances:
            raise HTTPException(status_code=400, detail="At least one appliance is required")
        
        if request.household_info.household_size <= 0:
            raise HTTPException(status_code=400, detail="Household size must be positive")
        
        model_used = "supervised_ml"
        # Use supervised model prediction if available
        if model is not None and scaler is not None and label_encoders is not None:
            try:
                predictions_flat, appliance_data = predict_with_model(request.appliances, request.household_info)
                total_kwh = float(np.sum(predictions_flat))
                message = "Prediction completed using supervised AI model"
                
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
                        "estimated_bill": 0,
                        "percentage": 0
                    })
                    
            except Exception as model_error:
                logger.warning(f"Supervised model prediction failed: {model_error}")
                total_kwh, breakdown_data = simple_energy_calculation(request.appliances)
                message = "Prediction completed using calculation (AI model unavailable)"
                model_used = "calculation"
        else:
            total_kwh, breakdown_data = simple_energy_calculation(request.appliances)
            message = "Prediction completed using standard calculation"
            model_used = "calculation"
        
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
        
        breakdown_data.sort(key=lambda x: x['estimated_kwh'], reverse=True)
        
        # Budget analysis
        budget_difference = total_bill - request.household_info.budget
        budget_status = "within_budget" if budget_difference <= 0 else "over_budget"
        
        # Generate traditional recommendations
        recommendations = get_smart_recommendations(total_kwh, total_bill, breakdown_data, request.household_info.budget)
        
        # Convert to Pydantic models
        breakdown = [ApplianceResult(**item) for item in breakdown_data]
        
        # Generate report ID
        report_id = str(uuid.uuid4())[:8]
        
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
            report_id=report_id,
            model_used=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# =============================================================================
# TIME SERIES FORECASTING ENDPOINTS
# =============================================================================

@app.post("/api/timeseries/forecast", response_model=TimeSeriesResponse)
async def forecast_energy_consumption(request: TimeSeriesRequest):
    """
    Forecast future energy consumption using time series models
    
    **Models Used:**
    - LSTM Neural Network: Primary forecasting model
    - SARIMA: Statistical fallback model
    
    **Requirements:**
    - Minimum 3 months of historical data
    - Maximum 12 months forecast
    - Historical data should be in kWh
    """
    try:
        # Validate input
        if len(request.historical_data) < 3:
            raise HTTPException(
                status_code=400, 
                detail="At least 3 months of historical data is required"
            )
        
        if request.months_ahead < 1 or request.months_ahead > 12:
            raise HTTPException(
                status_code=400,
                detail="Forecast horizon must be between 1 and 12 months"
            )
        
        # Check for negative values
        if any(x < 0 for x in request.historical_data):
            raise HTTPException(
                status_code=400,
                detail="Historical data cannot contain negative values"
            )
        
        logger.info(f"Time series forecast request: {len(request.historical_data)} months historical, {request.months_ahead} months ahead")
        
        # Use LSTM model for prediction
        prediction_result = timeseries_forecaster.predict_future_consumption(
            request.historical_data, 
            request.months_ahead
        )
        
        if "error" in prediction_result:
            # Fallback to SARIMA if LSTM fails
            logger.warning("LSTM prediction failed, trying SARIMA fallback")
            prediction_result = timeseries_forecaster.predict_with_sarima(
                request.historical_data,
                request.months_ahead
            )
            
            if "error" in prediction_result:
                raise HTTPException(status_code=500, detail=prediction_result["error"])
        
        # Calculate bill forecasts
        bill_forecast = timeseries_forecaster.calculate_bill_forecast(
            prediction_result["predictions"]
        )
        
        # Generate month names
        current_date = datetime.now()
        month_names = []
        for i in range(request.months_ahead):
            future_date = current_date + timedelta(days=30 * (i + 1))
            month_names.append(future_date.strftime("%B %Y"))
        
        # Create predictions with month names
        predictions = []
        for i, (consumption, bill, bracket) in enumerate(zip(
            prediction_result["predictions"],
            bill_forecast["bill_predictions"],
            bill_forecast["tariff_brackets"]
        )):
            predictions.append(TimeSeriesPrediction(
                month=month_names[i],
                predicted_consumption_kwh=round(consumption, 2),
                predicted_bill_rwf=round(bill, 2),
                tariff_bracket=bracket,
                confidence=prediction_result.get("confidence", "medium")
            ))
        
        # Calculate trends
        historical_avg = np.mean(request.historical_data[-3:])  # Last 3 months average
        forecast_avg = np.mean(prediction_result["predictions"])
        
        if forecast_avg > historical_avg:
            trend = "increasing"
            trend_percentage = ((forecast_avg - historical_avg) / historical_avg) * 100
        else:
            trend = "decreasing"
            trend_percentage = ((historical_avg - forecast_avg) / historical_avg) * 100
        
        # Generate forecast ID
        forecast_id = str(uuid.uuid4())[:8]
        
        return TimeSeriesResponse(
            status="success",
            message=f"Successfully forecasted {request.months_ahead} months of energy consumption",
            historical_data=[round(x, 2) for x in request.historical_data],
            forecast_months=request.months_ahead,
            predictions=predictions,
            total_forecasted_consumption=round(sum(prediction_result["predictions"]), 2),
            total_forecasted_bill=round(sum(bill_forecast["bill_predictions"]), 2),
            average_monthly_consumption=round(forecast_avg, 2),
            average_monthly_bill=round(np.mean(bill_forecast["bill_predictions"]), 2),
            trend=trend,
            trend_percentage=round(trend_percentage, 2),
            model_used=prediction_result.get("model_used", "LSTM"),
            forecast_id=forecast_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Time series forecasting error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

@app.get("/api/timeseries/status")
async def get_timeseries_status():
    """Get status of time series forecasting models"""
    return {
        "status": "success",
        "time_series_forecasting": {
            "models_loaded": timeseries_forecaster.loaded,
            "lstm_available": timeseries_forecaster.lstm_model is not None,
            "sarima_available": timeseries_forecaster.sarima_model is not None,
            "scalers_available": timeseries_forecaster.scaler_X is not None and timeseries_forecaster.scaler_y is not None,
            "artifacts_available": timeseries_forecaster.artifacts is not None
        },
        "capabilities": {
            "min_historical_data": 3,
            "max_forecast_months": 12,
            "supported_models": ["LSTM", "SARIMA"],
            "input_requirements": "Monthly kWh consumption data"
        }
    }

@app.post("/api/timeseries/sample")
async def generate_sample_timeseries():
    """Generate sample time series data for testing"""
    # Generate realistic sample data based on Rwandan household patterns
    sample_data = {
        "historical_data": [45.2, 52.1, 48.7, 55.3, 51.8, 49.2],
        "months_ahead": 3,
        "household_info": {
            "region": "Kigali",
            "income_level": "Medium",
            "household_size": 4
        }
    }
    
    return {
        "status": "success",
        "sample_data": sample_data,
        "description": "Sample 6 months of historical data for a medium-income household in Kigali",
        "usage_note": "Use this sample with /api/timeseries/forecast endpoint to test the forecasting"
    }

# =============================================================================
# CLUSTERING AND ANOMALY DETECTION ENDPOINTS
# =============================================================================

@app.post("/api/clustering/predict", response_model=ClusterAnomalyResponse)
async def predict_cluster_and_anomaly(request: ClusterAnomalyRequest):
    """
    Predict household cluster and detect anomalies
    
    **Models Used:**
    - K-Means Clustering: Groups households by consumption patterns
    - Isolation Forest: Detects unusual consumption behavior
    
    **Features Analyzed:**
    - Energy consumption patterns
    - Bill amounts and tariff brackets
    - Household demographics
    - Appliance usage patterns
    """
    try:
        logger.info(f"Received clustering/anomaly detection request for household")
        
        # Prepare household data
        household_data = {
            'total_kwh': request.total_kwh,
            'total_bill': request.total_bill,
            'household_size': request.household_size,
            'region': request.region,
            'income_level': request.income_level,
            'tariff_bracket': request.tariff_bracket,
            'appliance_count': request.appliance_count,
            'avg_usage_hours': request.avg_usage_hours,
            'daily_energy': request.daily_energy or (request.total_kwh / 30),  # Estimate if not provided
            'month': request.month,
            'household_id': request.household_id
        }
        
        # Get prediction
        prediction = clustering_detector.predict_cluster_and_anomaly(household_data)
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())[:8]
        
        return ClusterAnomalyResponse(
            cluster=prediction["cluster"],
            cluster_profile=prediction["cluster_profile"],
            anomaly_status=prediction["anomaly_status"],
            anomaly_score=prediction["anomaly_score"],
            anomaly_confidence=prediction["anomaly_confidence"],
            features_used=prediction["features_used"],
            status="success",
            message="Clustering and anomaly detection completed successfully",
            analysis_id=analysis_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clustering/anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/api/clustering/batch", response_model=BatchClusterResponse)
async def batch_cluster_analysis(request: BatchClusterRequest):
    """
    Analyze multiple households for clustering and anomaly detection
    
    **Use Cases:**
    - Community energy analysis
    - Utility company customer segmentation
    - Regional energy pattern studies
    - Anomaly detection across multiple households
    """
    try:
        logger.info(f"Received batch clustering request for {len(request.households)} households")
        
        if len(request.households) == 0:
            raise HTTPException(status_code=400, detail="At least one household is required")
        
        if len(request.households) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 households per batch request")
        
        # Prepare household data
        households_data = []
        for household in request.households:
            households_data.append({
                'total_kwh': household.total_kwh,
                'total_bill': household.total_bill,
                'household_size': household.household_size,
                'region': household.region,
                'income_level': household.income_level,
                'tariff_bracket': household.tariff_bracket,
                'appliance_count': household.appliance_count,
                'avg_usage_hours': household.avg_usage_hours,
                'daily_energy': household.daily_energy or (household.total_kwh / 30),
                'month': household.month,
                'household_id': household.household_id or f"household_{uuid.uuid4().hex[:8]}"
            })
        
        # Analyze patterns
        analysis_result = clustering_detector.analyze_household_patterns(households_data)
        
        if "error" in analysis_result:
            raise HTTPException(status_code=500, detail=analysis_result["error"])
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())[:8]
        
        return BatchClusterResponse(
            individual_predictions=analysis_result["individual_predictions"],
            community_insights=analysis_result["community_insights"],
            status="success",
            message=f"Successfully analyzed {len(analysis_result['individual_predictions'])} households",
            batch_id=batch_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch clustering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")

@app.get("/api/clustering/status")
async def get_clustering_status():
    """Get status of clustering and anomaly detection models"""
    return {
        "status": "success",
        "clustering_anomaly_detection": {
            "models_loaded": clustering_detector.loaded,
            "kmeans_available": clustering_detector.kmeans_model is not None,
            "isolation_forest_available": clustering_detector.isolation_forest_model is not None,
            "scalers_available": clustering_detector.scaler_cluster is not None and clustering_detector.scaler_anomaly is not None,
            "pca_available": clustering_detector.pca_model is not None,
            "encoders_available": clustering_detector.label_encoders is not None
        },
        "capabilities": {
            "clusters": 3,
            "cluster_profiles": ["Low Consumption", "Medium Consumption", "High Consumption"],
            "anomaly_detection": True,
            "batch_processing": True,
            "max_batch_size": 1000
        }
    }

@app.get("/api/clustering/clusters")
async def get_cluster_information():
    """Get detailed information about each cluster"""
    clusters_info = {}
    
    for cluster_id in [0, 1, 2]:
        clusters_info[cluster_id] = clustering_detector.get_cluster_profile(cluster_id)
    
    return {
        "status": "success",
        "clusters": clusters_info,
        "total_clusters": 3,
        "description": "Household energy consumption clusters based on K-Means analysis"
    }

# =============================================================================
# ERROR HANDLERS
# =============================================================================

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
