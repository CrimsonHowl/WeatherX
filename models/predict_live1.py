import smbus2, bme280, json, requests, os, numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Path from your Repo screenshot
MODEL_PATH = "models/saved_models/weatherx_multidistrict_lstm.h5"
BME_ADDR = 0x76

def run_pi_inference():
    print("🛰️ Starting Hardware-AI Fusion...")
    
    # 1. Read BME280
    bus = smbus2.SMBus(1)
    params = bme280.load_calibration_params(bus, BME_ADDR)
    sample = bme280.sample(bus, BME_ADDR, params)
    
    # 2. Prepare 24h Window (Mocking history, injecting live sensor)
    # Your model expects (1, 24, 8)
    input_data = np.zeros((1, 24, 8))
    input_data[0, 23, 1] = sample.temperature / 45.0 # AntiGravity Anchor
    
    # 3. Model Predict
    model = load_model(MODEL_PATH)
    raw_preds = model.predict(input_data)[0]
    
    # 4. Map to 39 Districts (Example mapping for 2 districts)
    districts = ["Chennai", "Coimbatore", "Madurai", "Salem"] # Add all 39
    forecasts = []
    for i, name in enumerate(districts):
        forecasts.append({
            "district": name, 
            "temp": round(float(raw_preds[i*8+1] * 45), 2),
            "lat": 13.08 if name == "Chennai" else 11.01, # Add real lat/lon
            "lon": 80.27 if name == "Chennai" else 76.95
        })

    # 5. Export
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed_station": {"temp": round(sample.temperature, 2), "source": "BME280-Hardware"},
        "forecasts": forecasts
    }
    
    with open('dashboard/latest_forecast.json', 'w') as f:
        json.dump(report, f, indent=4)
    print(" Hardware Prediction Uploaded.")

if __name__ == "__main__": run_pi_inference()
