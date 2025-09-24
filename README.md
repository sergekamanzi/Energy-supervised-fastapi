# Rwanda Energy Consumption Predictor API

A FastAPI-based machine learning service that predicts household energy consumption and electricity bills in Rwanda using neural networks.

##  Features

- **AI-Powered Predictions**: Multi-output neural network predicts kWh consumption, tariff brackets, and bill amounts
- **Rwanda-Specific**: Tailored for Rwanda's electricity tariff structure (0-20 kWh, 21-50 kWh, 50+ kWh)
- **Smart Recommendations**: Provides energy-saving tips and budget optimization advice
- **CORS Enabled**: Ready for web and mobile applications
- **RESTful API**: Clean JSON API with comprehensive documentation

##  Model Outputs

- **Energy Consumption**: Monthly kWh prediction
- **Tariff Bracket**: Automatic classification into Rwanda's tariff tiers
- **Bill Estimation**: Accurate bill calculation based on actual tariff rates
- **Appliance Breakdown**: Detailed consumption per appliance
- **Budget Analysis**: Compare predicted bills with your budget

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/sergekamanzi/rwanda-energy-predictor.git
cd rwanda-energy-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure model files are present**
```
rwanda_energy_model.h5
energy_scaler.pkl
label_encoders.pkl
feature_names.pkl
model_constants.pkl
```

```
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── rwanda_energy_model.h5 # Trained neural network
├── energy_scaler.pkl      # Feature scaler
├── label_encoders.pkl     # Categorical encoders
├── feature_names.pkl      # Model features
└── model_constants.pkl    # Tariff rates & constants
```



**Note**: This is a predictive model. Actual electricity bills may vary based on usage patterns and utility company policies.

For questions or support, please open an issue in the GitHub repository.
