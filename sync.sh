#!/bin/bash
# 🛰️ WeatherX Sentinel-Sync: RPi 3B Automation Script
# Optimized with "Pull-First" Resilience

PROJECT_DIR="/home/logz/WeatherX"
cd $PROJECT_DIR

echo "--------------------------------------------------"
echo "WeatherX: Checking for Cloud updates..."

# 1. PULL CHANGES (Ensures Pi has latest code from laptop/GitHub)
git pull origin main --rebase

echo "WeatherX: Initializing 39-District Neural Prediction..."

# 2. RUN THE AI ENGINE 
./vx_env/bin/python3 models/predict_live1.py

# 3. GITHUB CLOUD SYNC
echo "WeatherX: Synchronizing Data to Streamlit Cloud..."
git add dashboard/latest_forecast.json
git commit -m "Sentinel-Sync: Neural state updated at $(date)"
git push origin main

echo "--------------------------------------------------"
echo "WeatherX: System state synchronized. Done."
echo "--------------------------------------------------"
