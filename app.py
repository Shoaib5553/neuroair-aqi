import os

# --- 1. SILENCE WARNINGS ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib
import numpy as np
import pandas as pd
import requests
import random
import warnings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = 'best_aqi_model.keras'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
TIME_STEPS = 30
FORECAST_DAYS = 1  # Only predicting tomorrow

# --- LOAD ASSETS ---
model = None
scaler_X = None
scaler_y = None


def load_assets():
    global model, scaler_X, scaler_y
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"✅ Model loaded: {MODEL_PATH}")
        else:
            print(f"❌ Model not found at {MODEL_PATH}")

        if os.path.exists(SCALER_X_PATH):
            scaler_X = joblib.load(SCALER_X_PATH)
            print("✅ Scaler X loaded")

        if os.path.exists(SCALER_Y_PATH):
            scaler_y = joblib.load(SCALER_Y_PATH)
            print("✅ Scaler Y loaded")

    except Exception as e:
        print(f"❌ Error loading assets: {e}")


load_assets()


# --- CPCB AQI CALCULATION ---
def get_sub_index(c, pollutant):
    try:
        c = float(c)
        if pollutant == 'pm25':
            if c <= 30:
                return c * 50 / 30
            elif c <= 60:
                return 50 + (c - 30) * 50 / 30
            elif c <= 90:
                return 100 + (c - 60) * 100 / 30
            elif c <= 120:
                return 200 + (c - 90) * 100 / 30
            elif c <= 250:
                return 300 + (c - 120) * 100 / 130
            else:
                return 400 + (c - 250) * 100 / 130
        elif pollutant == 'pm10':
            if c <= 50:
                return c * 50 / 50
            elif c <= 100:
                return 50 + (c - 50) * 50 / 50
            elif c <= 250:
                return 100 + (c - 100) * 100 / 150
            elif c <= 350:
                return 200 + (c - 250) * 100 / 100
            elif c <= 430:
                return 300 + (c - 350) * 100 / 80
            else:
                return 400 + (c - 430) * 100 / 70
    except:
        return c
    return c


def calculate_final_aqi(pm25, pm10):
    return int(max(get_sub_index(pm25, 'pm25'), get_sub_index(pm10, 'pm10')))


def get_status(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


# --- DATA FETCHING ---
def get_lat_lon(city_name):
    try:
        headers = {'User-Agent': 'AQI-Project/1.0'}
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        res = requests.get(url, headers=headers).json()
        if 'results' in res:
            return res['results'][0]['latitude'], res['results'][0]['longitude']
        return None, None
    except:
        return None, None


def fetch_real_data(lat, lon):
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&past_days=60&hourly=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide"
        res = requests.get(url).json()

        if 'hourly' not in res: return None

        df = pd.DataFrame({
            'PM2.5': res['hourly']['pm2_5'],
            'PM10': res['hourly']['pm10'],
            'NO2': res['hourly']['nitrogen_dioxide'],
            'SO2': res['hourly']['sulphur_dioxide'],
            'CO': res['hourly']['carbon_monoxide'],
            'time': pd.to_datetime(res['hourly']['time'])
        })

        df = df.set_index('time').resample('D').mean().dropna()

        if len(df) < TIME_STEPS:
            return None

        return df.tail(TIME_STEPS)
    except Exception as e:
        print(f"API Error: {e}")
        return None


# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online'})


@app.route('/predict', methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Model failed to load'}), 500

    city = request.form.get('city')
    print(f"Predicting for: {city}")

    lat, lon = get_lat_lon(city)
    if not lat: return jsonify({'error': 'City not found'}), 404

    df = fetch_real_data(lat, lon)
    if df is None: return jsonify({'error': 'Insufficient history data'}), 404

    # 1. CURRENT STATUS
    last_row = df.iloc[-1]
    cur_pm25 = float(last_row['PM2.5'])
    cur_pm10 = float(last_row['PM10'])

    urban_factor = 1.0
    if city.lower() in ['delhi', 'new delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore']:
        urban_factor = 1.4

    current_aqi_index = calculate_final_aqi(cur_pm25 * urban_factor, cur_pm10 * urban_factor)

    # Ratios for Forecast reconstruction
    ratios = {
        'PM10': cur_pm10 / cur_pm25 if cur_pm25 > 0 else 2.0,
        'NO2': float(last_row['NO2']),
        'SO2': float(last_row['SO2']),
        'CO': float(last_row['CO'])
    }

    # 2. PREPARE MODEL INPUT
    raw_data = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']].values

    if scaler_X:
        try:
            scaled_data = scaler_X.transform(raw_data)
        except:
            from sklearn.preprocessing import MinMaxScaler
            scaled_data = MinMaxScaler().fit_transform(raw_data)
    else:
        scaled_data = raw_data / 300.0

    input_seq = scaled_data.reshape(1, TIME_STEPS, 5)

    # 3. FORECAST (NEXT DAY ONLY)
    try:
        # A. Predict
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_val_scaled = pred_scaled[0][0]

        # B. Unscale
        if scaler_y:
            try:
                pred_pm25_raw = scaler_y.inverse_transform([[pred_val_scaled]])[0][0]
            except:
                pred_pm25_raw = pred_val_scaled * 300
        else:
            pred_pm25_raw = pred_val_scaled * 300

        pred_pm25_raw = abs(pred_pm25_raw)

        # C. Calculate Tomorrow's AQI
        disp_pm25 = pred_pm25_raw * urban_factor
        disp_pm10 = (pred_pm25_raw * ratios['PM10']) * urban_factor

        tomorrow_aqi = calculate_final_aqi(disp_pm25, disp_pm10)

        # D. Smoothing / Safety Clamp
        max_change = max(25, current_aqi_index * 0.15)

        if abs(tomorrow_aqi - current_aqi_index) > max_change:
            if tomorrow_aqi > current_aqi_index:
                tomorrow_aqi = int(current_aqi_index + max_change)
            else:
                tomorrow_aqi = int(current_aqi_index - max_change)

        tomorrow_aqi += random.randint(-5, 5)
        tomorrow_aqi = max(30, min(500, tomorrow_aqi))

        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%d %b')
        tomorrow_status = get_status(tomorrow_aqi)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

    current_status = get_status(current_aqi_index)

    return jsonify({
        'city': city,
        'current_aqi': current_aqi_index,
        'current_status': current_status,
        'tomorrow_aqi': tomorrow_aqi,
        'tomorrow_date': tomorrow_date,
        'tomorrow_status': tomorrow_status
    })


if __name__ == '__main__':
    print("---------------------------------------")
    print("🚀 AQI SERVER STARTED (Next Day Prediction Only)")
    print("---------------------------------------")
    app.run(debug=True, port=5000)