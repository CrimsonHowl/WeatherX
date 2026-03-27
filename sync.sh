#!/bin/bash
# 🛰️ WeatherX Sentinel-Sync: RPi 3B Automation Script
# Optimized for V8.8 AntiGravity (Hardware-Integrated)

# 1. SET PROJECT PATH (Change 'logz' to your actual Pi username if different)
PROJECT_DIR="/home/logz/WeatherX"
cd $PROJECT_DIR

echo "--------------------------------------------------"
echo "WeatherX: Initializing 39-District Neural Prediction..."
echo "Timestamp: $(date)"

# 2. RUN THE AI ENGINE 
# We call the python binary INSIDE the virtual environment directly
# This ensures all libraries (TensorFlow/BME280) are found
./vx_env/bin/python3 models/predict_live1.py

# 3. WAIT FOR NEURAL PROCESSING (Brief pause to ensure file write is finished)
sleep 2

# 4. GITHUB CLOUD SYNC
echo "WeatherX: Synchronizing with Streamlit Cloud..."

# Adding the specific JSON file produced by the AI
git add dashboard/latest_forecast.json

# Commit with a timestamped message
git commit -m "Sentinel-Sync: Neural state updated at $(date)"

# Push to the main branch
git push origin main

echo "--------------------------------------------------"
echo "WeatherX: System state synchronized to GitHub."
echo "Sentinel Sleep Mode engaged. See you in 1 hour."
echo "--------------------------------------------------"
