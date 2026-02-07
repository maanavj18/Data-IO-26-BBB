from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import os

app = Flask(__name__)

# --- GLOBAL DATA LOADING ---
# We load these once on startup to avoid reading CSVs for every request.
# Ensure these files exist in your directory.
print("Loading datasets...")
try:
    # placeholders for your actual filenames
    METER_DATA_FILE = './master_dataset_combined.csv' 
    WEATHER_DATA_FILE = './energy_dataset/advanced_core/weather_data_hourly_2025.csv'
    BUILDING_DATA_FILE = './energy_dataset/advanced_core/building_metadata.csv'

    # Load data (using empty dataframes as placeholders if files don't exist for demo)
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
        # Ensure we can search by name (convert to string)
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
    """Robust ID cleaning."""
    return (pd.to_numeric(df[col_name], errors='coerce')
            .fillna(0)
            .astype(int)
            .astype(str))

def find_simscode_by_name(query_name):
    """
    Searches for a building SIMS code using a fuzzy string match on the name.
    Returns: (simscode, official_name) or (None, None)
    """
    if df_building.empty:
        return None, None
    
    # 1. Exact ID Match check
    if query_name.isdigit():
        clean_q = str(int(query_name))
        match = df_building[df_building['buildingnumber'] == clean_q]
        if not match.empty:
            return clean_q, match.iloc[0]['buildingname']

    # 2. Case-insensitive substring match
    matches = df_building[df_building['buildingname'].str.contains(query_name, case=False, na=False)]
    
    if matches.empty:
        return None, None
    
    # Return the first match
    return matches.iloc[0]['buildingnumber'], matches.iloc[0]['buildingname']

def generate_forecast(target_sims_code):
    """
    Core forecasting logic refactored for API usage.
    Returns a dictionary of results instead of plotting.
    """
    # 1. PREPARE METER DATA
    df_meter['simscode_clean'] = clean_ids(df_meter, 'simscode')
    
    # Clean target
    target_clean = str(int(float(target_sims_code))) if str(target_sims_code).replace('.','',1).isdigit() else str(target_sims_code)
    
    df_bldg = df_meter[
        (df_meter['simscode_clean'] == target_clean) & 
        (df_meter['utility'].isin(['ELECTRICAL_POWER', 'ELECTRICITY']))
    ].copy()
    
    if df_bldg.empty:
        return {"error": "No electricity data found for this building."}

    df_bldg['readingtime'] = pd.to_datetime(df_bldg['readingtime'])
    df_bldg = df_bldg.sort_values('readingtime')
    
    # Resample
    df_hourly = df_bldg.set_index('readingtime').resample('h')['readingvalue'].mean().reset_index()

    # 2. MERGE WEATHER
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_merged = pd.merge_asof(
        df_hourly, 
        df_weather[['date', 'temperature_2m', 'relative_humidity_2m']], 
        left_on='readingtime', 
        right_on='date', 
        direction='nearest'
    )

    # 3. FEATURES
    df = df_merged.copy()
    df['hour'] = df['readingtime'].dt.hour
    df['dayofweek'] = df['readingtime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['lag_24h'] = df['readingvalue'].shift(24)
    df = df.dropna()

    if df.empty:
        return {"error": "Insufficient historical data for lag features."}

    # 4. SPLIT & TRAIN
    split_point = len(df) - 24
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]

    features = ['hour', 'dayofweek', 'is_weekend', 'temperature_2m', 'lag_24h']
    target = 'readingvalue'

    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(train[features], train[target])

    # 5. PREDICT
    predictions = model.predict(test[features])
    
    # 6. FORMAT RESPONSE
    results = []
    for dt, pred, actual in zip(test['readingtime'], predictions, test[target]):
        results.append({
            "time": dt.strftime('%Y-%m-%d %H:%M:%S'),
            "predicted_kw": float(pred),
            "actual_kw": float(actual)
        })
        
    return {"status": "success", "forecast": results}

# --- API ENDPOINTS ---

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "OSU Energy Forecast API", 
        "usage": "/predict?building=Baker Hall"
    })

@app.route('/predict', methods=['GET'])
def predict():
    # Get building name from query parameter
    query_name = request.args.get('building')
    
    if not query_name:
        return jsonify({"error": "Please provide a 'building' parameter."}), 400

    print(f"Received request for: {query_name}")

    # 1. Lookup SIMS Code
    simscode, official_name = find_simscode_by_name(query_name)
    
    if not simscode:
        return jsonify({
            "error": "Building not found.", 
            "query": query_name
        }), 404

    # 2. Generate Forecast
    result = generate_forecast(simscode)
    
    if "error" in result:
        return jsonify(result), 500

    # 3. Return JSON
    response = {
        "query": query_name,
        "building_match": official_name,
        "simscode": simscode,
        "data": result['forecast']
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
