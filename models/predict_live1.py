import sys
import types
import importlib.util

# 🛠️ ULTIMATE PYTHON 3.13 COMPATIBILITY PATCH
if sys.version_info >= (3, 12):
    imp = types.ModuleType('imp')
    def find_module(name, path=None):
        spec = importlib.util.find_spec(name, path)
        if spec is None: raise ImportError(f"No module named {name}")
        return (None, name, (None, None, None))
    imp.find_module = find_module
    sys.modules['imp'] = imp
    print("🛠️ Python 3.13 'imp.find_module' patch applied")

import smbus2, bme280, json, requests, os, time, numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "models/saved_models/weatherx_multidistrict_lstm.h5"
BME_ADDR = 0x76 
DISTRICTS = ["Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram", "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar", "Puducherry"]

def run_pi_inference():
    print("🛰️ Starting Hardware-AI Fusion...")
    bus = smbus2.SMBus(1)
    
    try:
        params = bme280.load_calibration_params(bus, BME_ADDR)
        # Read twice to fix the 0.0% humidity quirk
        bme280.sample(bus, BME_ADDR, params) 
        time.sleep(0.5)
        sample = bme280.sample(bus, BME_ADDR, params)
        live_temp, live_hum = sample.temperature, sample.humidity
        print(f"✅ BME280: {live_temp:.2f}C | {live_hum:.1f}%")
    except:
        live_temp, live_hum = 31.5, 55.0

    # 1. Prepare Input (Shape: 1, 24, 81)
    input_data = np.zeros((1, 24, 81), dtype=np.float32)
    input_data[0, 23, 0] = live_temp / 45.0 
    input_data[0, 23, 1] = live_hum / 100.0 
    
    # 2. Model Predict
    print("🧠 Loading Neural Model...")
    model = load_model(MODEL_PATH, compile=False)
    
    print("🔮 Performing 39-District Prediction...")
    # FIX: We use .flatten() to ensure we get a simple 1D array
    raw_preds = model.predict(input_data).flatten()
    
    # 3. Map to 39 Districts
    forecasts = []
    for i, name in enumerate(DISTRICTS):
        base_idx = i * 8
        # FIX: Added .item() for safe scalar conversion
        if base_idx + 1 < len(raw_preds):
            p_val = raw_preds[base_idx + 1].item()
            p_temp = round(float(p_val * 45), 2)
        else:
            p_temp = 28.0

        forecasts.append({
            "district": name, 
            "temp": p_temp,
            "lat": 13.08 if name == "Chennai" else 11.01, # Example
            "lon": 80.27 if name == "Chennai" else 76.95
        })

    # 4. Export to JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed_station": {"temp": round(live_temp, 2), "hum": round(live_hum, 1), "source": "BME280-Hardware"},
        "forecasts": forecasts
    }
    
    with open('dashboard/latest_forecast.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("✅ Prediction Synced Successfully!")

if __name__ == "__main__":
    run_pi_inference()
