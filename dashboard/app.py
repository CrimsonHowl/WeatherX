import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests

# --- WeatherX: GitHub-Live Sync Configuration ---
st.set_page_config(
    page_title="WeatherX — Next-Gen Station 💠",
    page_icon="💠",
    layout="wide",
)

# 🔗 GITHUB CONFIG (Replace with your repo details for Cloud hosting)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/CrimsonHowl/weatherX/main/dashboard/latest_forecast.json"

# Regional Mapping
COASTAL = ["Chennai", "Chengalpattu", "Cuddalore", "Kancheepuram", "Kanniyakumari", "Mayiladuthurai", "Nagapattinam", "Puducherry", "Ramanathapuram", "Thanjavur", "Thoothukudi", "Tiruvallur", "Tiruvarur", "Villupuram"]
HILLY = ["Nilgiris", "Theni", "Dindigul"]

# Advanced CSS: Next-Gen Master UI
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
    .loading-text { font-family: 'Outfit', sans-serif; font-size: 24px; letter-spacing: 4px; color: #00d2ff; text-transform: uppercase; animation: blink 1.5s infinite; }
    @keyframes blink { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }

    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important; color: #ffffff !important; padding: 30px 15px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; font-weight: 600 !important;
    }
    .stButton > button:hover { border-color: #00d2ff !important; transform: translateY(-8px) scale(1.05) !important; box-shadow: 0 10px 30px rgba(0, 210, 255, 0.3) !important; }

    .badge-coastal { background: linear-gradient(90deg, #00d2ff, #3a7bd5); color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-left: 10px; }
    .badge-inland { background: linear-gradient(90deg, #f2994a, #f2c94c); color: #1a1a1a; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-left: 10px; }
    .badge-hilly { background: linear-gradient(90deg, #0ba360, #3cba92); color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-left: 10px; }

    .nebula-title { font-size: 64px; font-weight: 900; background: linear-gradient(90deg, #00d2ff, #a18cd1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; margin-bottom: 0px; }
    .sub-title { font-size: 16px; letter-spacing: 8px; color: rgba(255,255,255,0.3); text-transform: uppercase; margin-top: -10px; }

    .forecast-card-v8-8 { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 20px; padding: 22px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .glow-cyan { color: #00d2ff; font-weight: 700; font-size: 22px; }
    .glow-purple { color: #a18cd1; font-weight: 700; font-size: 22px; }
    </style>
    
    <div id="loading-overlay">
        <div class="pulse-loader"></div>
        <div class="loading-text">WeatherX: The Next Gen Weather Station</div>
    </div>
""", unsafe_allow_html=True)

# --- HYBRID DATA ENGINE: Local -> GitHub ---
@st.cache_data(ttl=60) # Cache for 1 min to prevent GitHub rate limits
def load_data():
    json_path = "dashboard/latest_forecast.json"
    # 1. Try Local File (for Raspberry Pi dev)
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    
    # 2. Try GitHub (for Cloud Hosting)
    try:
        response = requests.get(GITHUB_RAW_URL)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Cloud Sync Error: {e}")
    
    return None

# --- 🌀 THE DATA NORMALIZER (V8.9) ---
def normalize_data(data):
    if not data: return None
    # 1. Base Structure
    norm = {
        "timestamp": data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "seed_district": data.get('seed_district', 'Chennai'),
        "seed_station": data.get('seed_station', {"temp": 0.0, "source": "Station Offline"}),
        "districts": data.get('districts', {})
    }
    # 2. Handle "forecasts" list (New Pi Script)
    if 'forecasts' in data and not norm['districts']:
        for f in data['forecasts']:
            d_name = f.get('district', 'Unknown')
            if 'seed_district' not in data: norm['seed_district'] = d_name 
            f_safe = {
                "temp": f.get('temp', 28.0), "hum": f.get('hum', 55.0),
                "hour": 1, "rain": f.get('rain', 0.0), "lat": f.get('lat', 13.0), "lon": f.get('lon', 80.0)
            }
            norm['districts'][d_name] = {"current": f_safe, "forecast": [f_safe]}
    # 3. Validation
    if norm['seed_district'] not in norm['districts']:
        available = list(norm['districts'].keys())
        norm['seed_district'] = available[0] if available else 'Coimbatore'
    return norm

raw_data = load_data()
data = normalize_data(raw_data)

if not data:
    st.title("WeatherX: The Next Gen Weather Station")
    st.info("💠 Neural Engine Waiting for Data. Initializing Highland-Coastal Sync...")
    st.stop()

# Selection Management
if 'selected' not in st.session_state:
    st.session_state.selected = data['seed_district']

# --- DASHBOARD CONTENT ---
st.markdown("<div style='animation: fadeIn 0.8s ease-out;'>", unsafe_allow_html=True)
c_title, c_badge = st.columns([3, 1])
with c_title:
    st.markdown("<h1 class='nebula-title'>WEATHERX</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>The Next Gen Weather Station</p>", unsafe_allow_html=True)
with c_badge:
    st.markdown(f"""
        <div style="background: rgba(0, 210, 255, 0.05); border: 1px solid rgba(0, 210, 255, 0.2); border-radius: 15px; padding: 15px; text-align: right;">
            <div style="color: #00d2ff; font-size: 10px; font-weight: 800; letter-spacing: 2px;">NODE STATUS</div>
            <div style="font-size: 24px; font-weight: 900; color: #ffffff;">{data['seed_station'].get('temp', 0):.1f}°C</div>
            <div style="color: rgba(255,255,255,0.4); font-size: 10px;">{data['seed_station'].get('source', 'Unknown')}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown(f"<p style='color:#00ff7f; font-weight:600; margin-top:-15px;'>📍 PRIMARY NODE: {data['seed_district']} | {data['timestamp']}</p>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
hd = data['districts'][data['seed_district']]['current']
m1.metric("TEMP", f"{hd.get('temp', 0):.2f}°C", "LIVE")
m2.metric("HUM", f"{hd.get('hum', 0):.1f}%", "STABLE")
m3.metric("PRES", "1011 mb")
m4.metric("RAIN", f"{hd.get('rain', 0):.2f}mm")

st.markdown("---")
col_info, col_grid = st.columns([1.7, 2.3])

with col_info:
    sel = st.session_state.selected
    s_data = data['districts'][sel]
    if sel in COASTAL: tag_class, tag_text = "badge-coastal", "COASTAL DISTRICT"
    elif sel in HILLY: tag_class, tag_text = "badge-hilly", "HIGHLAND REGION"
    else: tag_class, tag_text = "badge-inland", "INLAND DISTRICT"
    
    st.markdown(f"## 📍 {sel} <span class='{tag_class}'>{tag_text}</span>", unsafe_allow_html=True)
    for f in s_data['forecast']:
        st.markdown(f"""
            <div class="forecast-card-v8-8">
                <div style="flex:0.5; color:rgba(255,255,255,0.3); font-size:10px; letter-spacing:1px; text-transform:uppercase;">+{f['hour']}H Neural</div>
                <div class="glow-cyan">{f['temp']:.2f}°C</div>
                <div class="glow-purple">{f['hum']:.1f}%</div>
                <div style="text-align:right; font-size:12px; color:rgba(255,255,255,0.5);">
                    <div style="color:#00ff7f;">1011 mb</div>
                    <div>{f.get('rain', 0.0):.2f} mm</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

with col_grid:
    st.subheader("🌐 STATE-WIDE SENSOR GRID")
    dn = sorted(list(data['districts'].keys()))
    if data['seed_district'] in dn:
        dn.remove(data['seed_district'])
        dn.insert(0, data['seed_district'])
    
    n_cols = 3
    for i in range(0, len(dn), n_cols):
        row = dn[i : i + n_cols]
        cols = st.columns(n_cols)
        for j, name in enumerate(row):
            t = data['districts'][name]['current']['temp']
            if cols[j].button(f"📍 {name}\n\n{t:.1f}°C", key=f"btn_{name}", use_container_width=True):
                st.session_state.selected = name
                st.rerun()

st.markdown("---")
st.subheader(f"📊 {sel}: Neural Trend Trends")
df_f = pd.DataFrame(data['districts'][sel]['forecast'])
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_f['hour'], y=df_f['temp'], name='Temp (°C)', line=dict(color='#00d2ff', width=6), mode='lines+markers'))
fig.add_trace(go.Scatter(x=df_f['hour'], y=df_f['hum'], name='Hum (%)', line=dict(color='#a18cd1', width=3, dash='dot'), mode='lines+markers', yaxis='y2'))
fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                  yaxis=dict(title="Temperature (°C)", title_font=dict(color="#00d2ff")), 
                  yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
                  legend=dict(orientation="h", x=1, xanchor="right", y=1.1))
st.plotly_chart(fig, use_container_width=True)

st.caption("WeatherX V8.8 GitHub-Live Sync • Tamil Nadu Project Final")
st.markdown("</div>", unsafe_allow_html=True)
