from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import requests
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
COLUMBUS_LAT = 39.9612
COLUMBUS_LON = -83.0032

# --- GLOBAL DATA LOADING ---
print("Loading datasets...")
try:
    # placeholders for your actual filenames
    METER_DATA_FILE = './master_dataset_combined.csv' 
    WEATHER_DATA_FILE = './energy_dataset/advanced_core/weather_data_hourly_2025.csv'
    BUILDING_DATA_FILE = './energy_dataset/advanced_core/building_metadata.csv'

    if os.path.exists(METER_DATA_FILE):
        df_meter = pd.read_csv(METER_DATA_FILE)
    else:
        print(f"WARNING: {METER_DATA_FILE} not found. API will fail on prediction.")
        df_meter = pd.DataFrame()

    if os.path.exists(WEATHER_DATA_FILE):
        df_weather = pd.read_csv(WEATHER_DATA_FILE)
    else:
        print(f"WARNING: {WEATHER_DATA_FILE} not found.")
        df_weather = pd.DataFrame()

    if os.path.exists(BUILDING_DATA_FILE):
        df_building = pd.read_csv(BUILDING_DATA_FILE)
        df_building['buildingname'] = df_building['buildingname'].astype(str)
        df_building['buildingnumber'] = pd.to_numeric(df_building['buildingnumber'], errors='coerce').fillna(0).astype(int).astype(str)
    else:
        print(f"WARNING: {BUILDING_DATA_FILE} not found. Name lookup will not work.")
        df_building = pd.DataFrame()
        
    print("Data loaded successfully.")

except Exception as e:
    print(f"Error loading data: {e}")

# --- HELPER FUNCTIONS ---

def clean_ids(df, col_name):
    return (pd.to_numeric(df[col_name], errors='coerce')
            .fillna(0)
            .astype(int)
            .astype(str))

def find_simscode_by_name(query_name):
    if df_building.empty:
        return None, None
    
    if query_name.isdigit():
        clean_q = str(int(query_name))
        match = df_building[df_building['buildingnumber'] == clean_q]
        if not match.empty:
            return clean_q, match.iloc[0]['buildingname']

    matches = df_building[df_building['buildingname'].str.contains(query_name, case=False, na=False)]
    if matches.empty:
        return None, None
    return matches.iloc[0]['buildingnumber'], matches.iloc[0]['buildingname']

def get_live_weather_forecast():
    """
    Fetches the next 24 hours of weather for Columbus, OH from Open-Meteo.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": COLUMBUS_LAT,
        "longitude": COLUMBUS_LON,
        "hourly": "temperature_2m,relative_humidity_2m",
        "forecast_days": 2, # Request 2 days to ensure we cover the next 24h fully
        "temperature_unit": "fahrenheit"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        hourly = data['hourly']
        df_forecast = pd.DataFrame({
            'date': pd.to_datetime(hourly['time']),
            'temperature_2m': hourly['temperature_2m'],
            'relative_humidity_2m': hourly['relative_humidity_2m']
        })
        
        # Filter for strictly the next 24 hours from NOW
        now = datetime.now()
        end_time = now + timedelta(hours=24)
        mask = (df_forecast['date'] >= now) & (df_forecast['date'] <= end_time)
        return df_forecast.loc[mask].head(24).reset_index(drop=True)
        
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None

def generate_forecast(target_sims_code):
    # 1. PREPARE HISTORICAL DATA (For Training)
    df_meter['simscode_clean'] = clean_ids(df_meter, 'simscode')
    target_clean = str(int(float(target_sims_code))) if str(target_sims_code).replace('.','',1).isdigit() else str(target_sims_code)
    
    df_bldg = df_meter[
        (df_meter['simscode_clean'] == target_clean) & 
        (df_meter['utility'].isin(['ELECTRICAL_POWER', 'ELECTRICITY']))
    ].copy()
    
    if df_bldg.empty:
        return {"error": "No electricity data found for this building."}

    df_bldg['readingtime'] = pd.to_datetime(df_bldg['readingtime'])
    df_bldg = df_bldg.sort_values('readingtime')
    df_hourly = df_bldg.set_index('readingtime').resample('h')['readingvalue'].mean().reset_index()

    # Merge Historical Weather for Training
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_train_merged = pd.merge_asof(
        df_hourly, 
        df_weather[['date', 'temperature_2m', 'relative_humidity_2m']], 
        left_on='readingtime', 
        right_on='date', 
        direction='nearest'
    )
    
    # Feature Engineering (Training)
    df_train = df_train_merged.copy()
    df_train['hour'] = df_train['readingtime'].dt.hour
    df_train['dayofweek'] = df_train['readingtime'].dt.dayofweek
    df_train['is_weekend'] = df_train['dayofweek'].isin([5, 6]).astype(int)
    df_train['lag_24h'] = df_train['readingvalue'].shift(24)
    df_train = df_train.dropna()

    if df_train.empty:
        return {"error": "Insufficient historical data for training."}

    # Train Model
    features = ['hour', 'dayofweek', 'is_weekend', 'temperature_2m', 'lag_24h']
    target = 'readingvalue'
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(df_train[features], df_train[target])

    # 2. PREPARE FUTURE DATA (For Prediction)
    df_future_weather = get_live_weather_forecast()
    
    if df_future_weather is None or df_future_weather.empty:
        return {"error": "Could not fetch live weather forecast."}

    # Create Future Features
    df_future = df_future_weather.copy()
    df_future['hour'] = df_future['date'].dt.hour
    df_future['dayofweek'] = df_future['date'].dt.dayofweek
    df_future['is_weekend'] = df_future['dayofweek'].isin([5, 6]).astype(int)
    
    # KEY STEP: Get the LAST 24 hours of known energy data to serve as lag features
    # Note: In a real app, you'd want live meter data. For the hackathon, we assume
    # the CSV contains "recent" data and grab the very last 24 entries.
    last_24h_energy = df_hourly['readingvalue'].tail(24).values
    
    if len(last_24h_energy) < 24:
        return {"error": "Not enough historical meter data to predict the future (need last 24h)."}
    
    # Assign lags (The energy at T-24 relative to the prediction time)
    # If predicting 1 PM tomorrow, we need 1 PM today.
    # We map the last 24h of actuals to the next 24h of predictions.
    df_future['lag_24h'] = last_24h_energy

    # 3. PREDICT
    predictions = model.predict(df_future[features])
    
    # 4. FORMAT RESPONSE
    results = []
    for dt, pred in zip(df_future['date'], predictions):
        results.append({
            "time": dt.strftime('%Y-%m-%d %H:%M:%S'),
            "predicted_kw": float(pred),
            "weather_temp_f": float(df_future.loc[df_future['date'] == dt, 'temperature_2m'].values[0])
        })
        
    return {"status": "success", "forecast": results}

# --- API ENDPOINTS ---

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "OSU Energy Forecast API (Live Weather)", 
        "usage": "/predict?building=Baker Hall"
    })

@app.route('/predict', methods=['GET'])
def predict():
    query_name = request.args.get('building')
    print(query_name)
    if not query_name:
        return jsonify({"error": "Please provide a 'building' parameter."}), 400

    simscode, official_name = find_simscode_by_name(query_name)
    if not simscode:
        return jsonify({"error": "Building not found.", "query": query_name}), 404

    result = generate_forecast(simscode)
    if "error" in result:
        return jsonify(result), 500

    return jsonify({
        "query": query_name,
        "building_match": official_name,
        "simscode": simscode,
        "data": result['forecast']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    