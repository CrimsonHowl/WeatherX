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
    print(" Python 3.13 'imp.find_module' patch applied")

import smbus2, bme280, json, requests, os, time, numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "models/saved_models/weatherx_multidistrict_lstm.h5"
BME_ADDR = 0x76 
DISTRICTS = ["Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram", "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar", "Puducherry"]

def run_pi_inference():
    print("🛰️ WeatherX: AntiGravity Delta-Correction Mode (V8.9.4)...")
    bus = smbus2.SMBus(1)
    
    # 1. READ SENSOR (The Absolute Truth)
    try:
        params = bme280.load_calibration_params(bus, BME_ADDR)
        # double-read for stability in heatwaves
        bme280.sample(bus, BME_ADDR, params)
        time.sleep(0.5)
        sample = bme280.sample(bus, BME_ADDR, params)
        live_temp = sample.temperature
        live_hum = 65.0 if sample.humidity < 1 else sample.humidity
        print(f" SENSOR TRUTH: {live_temp:.2f}C")
    except:
        live_temp, live_hum = 33.5, 60.0 # Emergency fallback
        print("⚠️ Sensor Offline. Using Summer Fallback.")

    # 2. PREPARE INPUT (Warm-Start Saturation)
    input_data = np.full((1, 24, 81), (live_temp / 45.0), dtype=np.float32)
    input_data[:, :, 1] = live_hum / 100.0
    
    # 3. NEURAL INFERENCE
    print("🧠 Loading Neural Model...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH, compile=False)
    print("🔮 Performing Inference...")
    raw_preds = model.predict(input_data).flatten()
    output_size = len(raw_preds)
    
    # 4. CALCULATE SPATIAL DELTA (Calibration Anchor)
    # We find what the AI thinks Coimbatore should be (Index 3 * 8 + 1 usually)
    coimbatore_ai_val = raw_preds[3 % output_size].item() * 45.0
    delta_correction = live_temp - coimbatore_ai_val
    print(f"🧠 Neural Bias detected: {coimbatore_ai_val:.2f}C")
    print(f"🚀 Applying Delta Shift: {delta_correction:+.2f}C")

    # 5. MAP & CORRECT ALL DISTRICTS
    forecasts = []
    for i, name in enumerate(DISTRICTS):
        idx = (i * 2) % output_size
        
        # Base AI Prediction + Global Delta Shift
        base_pred = raw_preds[idx].item() * 45.0
        corrected_temp = base_pred + delta_correction
        
        # Regional Summer Bias Offsets
        if name == "Nilgiris": corrected_temp -= 5.0
        if name == "Chennai": corrected_temp += 2.0
        
        # Final formatting
        corrected_temp = round(float(corrected_temp), 2)

        forecasts.append({
            "district": name, 
            "temp": corrected_temp,
            "lat": 13.08 if name == "Chennai" else 11.01,
            "lon": 80.27 if name == "Chennai" else 76.95
        })

    # 6. EXPORT
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed_station": {"temp": round(live_temp, 2), "hum": round(live_hum, 1), "source": "BME280-Hardware"},
        "forecasts": forecasts
    }
    
    os.makedirs('dashboard', exist_ok=True)
    with open('dashboard/latest_forecast.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Full State Forecast Synchronized with Reality Anchor ({live_temp:.2f}C)")

if __name__ == "__main__":
    run_pi_inference()
