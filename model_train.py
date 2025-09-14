import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Convert Excel to CSV


# Load the CSV dataset
df = pd.read_csv('Custom_Crops_yield_Historical_Dataset.csv')

# Handle missing values
df = df.dropna()

# Define features (X) and target (y)
feature_columns = [
    'Year', 'State Name', 'Dist Name', 'Crop', 'Area_ha',
    
    'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm', 
    'Wind_Speed_m_s'
]

target_column = 'Yield_kg_per_ha'

# Select available columns from your dataset
available_features = [col for col in feature_columns if col in df.columns]

X = df[available_features]
y = df[target_column]

# Handle categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Label encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Save the model and preprocessors
with open('crop_yield_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(available_features, f)

with open('categorical_columns.pkl', 'wb') as f:
    pickle.dump(categorical_columns, f)

with open('numerical_columns.pkl', 'wb') as f:
    pickle.dump(numerical_columns, f)

# Function to make predictions on new data from farmer input + API data
def predict_yield_from_farmer_input(state_name, district_name, crop, area_ha, weather_api_data, soil_api_data):
    """
    Make yield prediction using farmer input + real-time API data
    
    Parameters:
    - state_name: str (from farmer)
    - district_name: str (from farmer) 
    - crop: str (from farmer)
    - area_ha: float (from farmer)
    - weather_api_data: dict with keys ['temperature', 'humidity', 'rainfall', 'wind_speed']
    - soil_api_data: dict with keys ['ph']
    
    Returns predicted yield in kg/ha
    """
    # Load saved model and preprocessors
    with open('crop_yield_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('categorical_columns.pkl', 'rb') as f:
        cat_cols = pickle.load(f)
    
    with open('numerical_columns.pkl', 'rb') as f:
        num_cols = pickle.load(f)
    
    # Prepare input data dictionary
    new_data_dict = {
        'State Name': state_name,
        'Dist Name': district_name,
        'Year': 2025,  # Current year
        'Crop': crop,
        'pH': soil_api_data['ph'],
        'Temperature_C': weather_api_data['temperature'],
        'Humidity_%': weather_api_data['humidity'],
        'Area_ha': area_ha,
        'Rainfall_mm': weather_api_data['rainfall'],
        'Wind_Speed_m_s': weather_api_data['wind_speed']
    }
    
    # Create dataframe from input
    new_df = pd.DataFrame([new_data_dict])
    
    # Apply same preprocessing
    for col in cat_cols:
        if col in new_df.columns:
            new_df[col] = encoders[col].transform(new_df[col])
    
    # Scale numerical features (use the same columns as during training)
    new_df[num_cols] = scaler.transform(new_df[num_cols])
    
    # Make prediction
    prediction = model.predict(new_df[features])
    return prediction[0]

# Example usage:
# farmer_input = {
#     'state': 'Chhattisgarh',
#     'district': 'Durg', 
#     'crop': 'rice',
#     'area': 2.5
# }
# 
# weather_data = {
#     'temperature': 25.0,
#     'humidity': 80.0,
#     'rainfall': 1200.0,
#     'wind_speed': 2.0
# }
# 
# soil_data = {
#     'ph': 6.5
# }
# 
# predicted_yield = predict_yield_from_farmer_input(
#     farmer_input['state'], 
#     farmer_input['district'],
#     farmer_input['crop'],
#     farmer_input['area'],
#     weather_data,
#     soil_data
# )

print("Model saved successfully!")
if __name__ == "__main__":
    # Farmer input (manually provided for testing)
    farmer_input = {
        'state': 'Bihar',
        'district': 'kolkata', 
        'crop': 'maize',
        'area': 2.5
    }
    
    # Weather data (normally from weather API)
    weather_data = {
        'temperature': 25.0,
        'humidity': 80.0,
        'rainfall': 1200.0,
        'wind_speed': 2.0
    }
    
    # Soil data (normally from soil API)
    soil_data = {
        'ph': 6.5
    }
    
    # Make prediction
    try:
        predicted_yield = predict_yield_from_farmer_input(
            farmer_input['state'], 
            farmer_input['district'],
            farmer_input['crop'],
            farmer_input['area'],
            weather_data,
            soil_data
        )
        
        print(f"Farmer Details:")
        print(f"  State: {farmer_input['state']}")
        print(f"  District: {farmer_input['district']}")
        print(f"  Crop: {farmer_input['crop']}")
        print(f"  Area: {farmer_input['area']} hectares")
        print(f"\nWeather Conditions:")
        print(f"  Temperature: {weather_data['temperature']}Ã‚Â°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Rainfall: {weather_data['rainfall']}mm")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"\nSoil Conditions:")
        print(f"  pH: {soil_data['ph']}")
        print(f"\n{'='*50}")
        print(f"PREDICTED YIELD: {predicted_yield:.2f} kg/ha")
        print(f"TOTAL EXPECTED PRODUCTION: {predicted_yield * farmer_input['area']:.2f} kg")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Make sure you have trained the model first!")