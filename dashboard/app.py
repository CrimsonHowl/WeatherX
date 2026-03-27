import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests

# --- WeatherX: Configuration ---
st.set_page_config(
    page_title="WeatherX — Next-Gen Station 💠",
    page_icon="💠",
    layout="wide",
)

# 🔗 GITHUB CONFIG
GITHUB_RAW_URL = "https://raw.githubusercontent.com/logan-weatherx/weatherX/main/dashboard/latest_forecast.json"

# Regional Mapping
COASTAL = ["Chennai", "Chengalpattu", "Cuddalore", "Kancheepuram", "Kanniyakumari", "Mayiladuthurai", "Nagapattinam", "Puducherry", "Ramanathapuram", "Thanjavur", "Thoothukudi", "Tiruvallur", "Tiruvarur", "Villupuram"]
HILLY = ["Nilgiris", "Theni", "Dindigul"]

# Advanced Master UI CSS
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #010108 !important;
        background: radial-gradient(circle at top right, #050525, #010108) !important;
        color: #f0f0f0 !important;
    }
    header, footer, #MainMenu { visibility: hidden !important; height: 0 !important; }

    #loading-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #010108;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        z-index: 10000; animation: fadeOut 1.5s ease-out 1.5s forwards; pointer-events: none;
    }
    @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; visibility: hidden; } }
    .pulse-loader { width: 60px; height: 60px; border: 4px solid #00d2ff; border-radius: 50%; border-top-color: transparent; animation: spin 1s linear infinite; margin-bottom: 20px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important; color: #ffffff !important; padding: 20px 10px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; font-weight: 600 !important;
    }
    .stButton > button:hover { border-color: #00d2ff !important; transform: translateY(-5px) !important; box-shadow: 0 10px 30px rgba(0, 210, 255, 0.3) !important; }

    .badge-coastal { background: linear-gradient(90deg, #00d2ff, #3a7bd5); color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; }
    .badge-inland { background: linear-gradient(90deg, #f2994a, #f2c94c); color: #1a1a1a; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; }
    .badge-hilly { background: linear-gradient(90deg, #0ba360, #3cba92); color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; }

    .nebula-title { font-size: 64px; font-weight: 900; background: linear-gradient(90deg, #00d2ff, #a18cd1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; margin-bottom: 0px; }
    .sub-title { font-size: 16px; letter-spacing: 8px; color: rgba(255,255,255,0.3); text-transform: uppercase; margin-top: -10px; }

    .forecast-card-v8-8 { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 20px; padding: 22px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .glow-cyan { color: #00d2ff; font-weight: 700; font-size: 22px; }
    </style>
    
    <div id="loading-overlay">
        <div class="pulse-loader"></div>
        <div style="font-family: sans-serif; letter-spacing: 4px; color: #00d2ff;">WEATHERX: SYNCING NEURAL ENGINE</div>
    </div>
""", unsafe_allow_html=True)

# --- HYBRID DATA ENGINE ---
@st.cache_data(ttl=30)
def load_data():
    try:
        # Fetch from GitHub
        response = requests.get(GITHUB_RAW_URL)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Sync Error: {e}")
    return None

data = load_data()

# --- SAFETY CHECK & RESILIENCY ---
if not data or 'forecasts' not in data:
    st.info("💠 Neural Engine Initializing... Please ensure Raspberry Pi has performed the first sync.")
    st.stop()

# Convert the flat list from RPi into a Dictionary for easy lookup
dist_dict = {item['district']: item for item in data['forecasts']}
seed_info = data.get('seed_station', {"temp": 0, "hum": 0, "source": "BME280-Hardware"})
timestamp = data.get('timestamp', 'Unknown')

if 'selected' not in st.session_state:
    st.session_state.selected = "Coimbatore" # Default selection

# --- DASHBOARD CONTENT ---
st.markdown("<h1 class='nebula-title'>WEATHERX</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>The Next Gen Weather Station</p>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#00ff7f; font-weight:600; margin-top:-15px;'>📍 PRIMARY NODE: Coimbatore | {timestamp}</p>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("TEMP (SEED)", f"{seed_info.get('temp', 0):.2f}°C", "LIVE")
m2.metric("HUM (SEED)", f"{seed_info.get('hum', 0):.1f}%", "STABLE")
m3.metric("PRES", "1011 mb")
m4.metric("RAIN", "0.00 mm")

st.markdown("---")
col_info, col_grid = st.columns([1.5, 2.5])

with col_info:
    sel = st.session_state.selected
    # Get selected data safely
    s_data = dist_dict.get(sel, {"temp": 0.0, "hum": 0.0, "rain": 0.0})
    
    if sel in COASTAL: tag_class, tag_text = "badge-coastal", "COASTAL DISTRICT"
    elif sel in HILLY: tag_class, tag_text = "badge-hilly", "HIGHLAND REGION"
    else: tag_class, tag_text = "badge-inland", "INLAND DISTRICT"
    
    st.markdown(f"## 📍 {sel} <span class='{tag_class}'>{tag_text}</span>", unsafe_allow_html=True)
    
    # Display the Prediction
    st.markdown(f"""
        <div class="forecast-card-v8-8">
            <div style="color:rgba(255,255,255,0.3); font-size:10px; letter-spacing:1px;">+6H NEURAL</div>
            <div class="glow-cyan">{s_data['temp']:.2f}°C</div>
            <div style="color:#a18cd1; font-weight:700;">{s_data.get('hum', 55):.1f}%</div>
            <div style="text-align:right; font-size:12px; color:rgba(255,255,255,0.5);">
                <div style="color:#00ff7f;">PRED.RAIN</div>
                <div>{s_data.get('rain', 0.0):.2f} mm</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.info("The AntiGravity AI uses the Seed Station as a thermal anchor to adjust this regional prediction.")

with col_grid:
    st.subheader("🌐 STATE-WIDE SENSOR GRID")
    dn = sorted(list(dist_dict.keys()))
    
    n_cols = 3
    for i in range(0, len(dn), n_cols):
        row = dn[i : i + n_cols]
        cols = st.columns(n_cols)
        for j, name in enumerate(row):
            t = dist_dict[name]['temp']
            if cols[j].button(f"📍 {name}\n\n{t:.1f}°C", key=f"btn_{name}", use_container_width=True):
                st.session_state.selected = name
                st.rerun()

st.markdown("---")
st.caption("WeatherX V8.8 AntiGravity • AI-Driven Tamil Nadu Station • Logic: BME280 Hardware + LSTM")
