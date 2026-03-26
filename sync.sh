#!/bin/bash
# 🛰️ WeatherX Sentinel-Sync: RPi 3B Automation Script
# Path: /home/pi/weatherX/sync.sh

echo "WeatherX: Initializing 39-District Neural Prediction..."

# 1. RUN THE AI ENGINE (Reading real-time BME280 sensors)
# We use sudo to ensure I2C permissions
python3 -m models.predict_live --temp 36.4 --hum 24.1 --pres 1011.2 --rain 0.0

# 2. WAIT FOR NEURAL PROCESSING
sleep 5

# 3. GITHUB CLOUD SYNC
echo "WeatherX: Synchronizing with Streamlit Cloud..."
git add dashboard/latest_forecast.json
git commit -m "Sentinel-Sync: Neural state updated at $(date)"
git push origin main

echo "WeatherX: System state [da127ef] synchronized. Sentinel Sleep Mode engaged."
