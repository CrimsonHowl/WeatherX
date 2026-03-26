import os
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.inference import WeatherXInferencer

# Re-use config from preprocess (ideally these would be in a shared config.py)
TARGET_COLS = ["precipitation", "temperature_2m", "relative_humidity_2m"]
SEED_DISTRICT = "Coimbatore"

def engineer_single_window(df, seed_district, lag_map, feature_cols, district_order, overrides=None):
    """
    Applies the same feature engineering logic as preprocess.py to a single window.
    Supports AntiGravity Temporal Shift to align history with live sensors.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # --- AntiGravity V6.1 Vector Delta Projection ---
    # Step 1: Calculate the Reality Gap at the Seed Station (Coimbatore BME280 vs API Baseline)
    # Step 2: Propagate that gap GLOBALLY to all 39 districts (Spatial Sensor Fusion)
    # Step 3: Pin Coimbatore's final row to the exact sensor reading (local precision)
    if overrides:
        mask = (df["district"] == seed_district)
        if mask.any():
            seed_data = df[mask].sort_values("timestamp")
            
            # --- STEP 1: Calculate Deltas (Reality Gap) ---
            t_delta = 0.0
            h_delta = 0.0
            p_delta = 0.0

            if "temp" in overrides and overrides["temp"] is not None:
                api_temp = seed_data["temperature_2m"].iloc[-1]
                t_delta = overrides["temp"] - api_temp

            if "hum" in overrides and overrides["hum"] is not None:
                api_hum = seed_data["relative_humidity_2m"].iloc[-1]
                h_delta = overrides["hum"] - api_hum

            if "pres" in overrides and overrides["pres"] is not None:
                api_pres = seed_data["surface_pressure"].iloc[-1]
                p_delta = overrides["pres"] - api_pres

            # --- STEP 2: Apply Delta GLOBALLY to all districts ---
            if t_delta != 0.0:
                df["temperature_2m"] = df["temperature_2m"] + t_delta
            if h_delta != 0.0:
                df["relative_humidity_2m"] = (df["relative_humidity_2m"] + h_delta).clip(0, 100)
            if p_delta != 0.0:
                df["surface_pressure"] = df["surface_pressure"] + p_delta

            print(f"  [V6.1 Vector Delta] Global Sync: ΔT={t_delta:+.1f}°C, ΔH={h_delta:+.1f}%, ΔP={p_delta:+.1f}hPa")

            # --- STEP 3: Pin Coimbatore's final row to exact sensor reading ---
            last_idx = df[mask].sort_values("timestamp").index[-1]
            if "temp" in overrides and overrides["temp"] is not None:
                df.at[last_idx, "temperature_2m"] = overrides["temp"]
            if "hum" in overrides and overrides["hum"] is not None:
                df.at[last_idx, "relative_humidity_2m"] = overrides["hum"]
            if "pres" in overrides and overrides["pres"] is not None:
                df.at[last_idx, "surface_pressure"] = overrides["pres"]
            if "rain" in overrides and overrides["rain"] is not None:
                df.at[last_idx, "precipitation"] = overrides["rain"]
            if "gtemp" in overrides and overrides["gtemp"] is not None:
                df.at[last_idx, "gtemp"] = overrides["gtemp"]

    # 0. District Mapping for ID feature
    dist_to_id = {d: i for i, d in enumerate(district_order)} 
    df["district_id"] = df["district"].map(dist_to_id)

    # 1. Cyclical Time
    hour = df["timestamp"].dt.hour
    doy  = df["timestamp"].dt.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_doy"]  = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"]  = np.cos(2 * np.pi * doy / 365)

    processed_parts = []
    for district, grp in df.groupby("district"):
        grp = grp.sort_values("timestamp").copy()
        # Pressure tendency
        grp["pressure_tendency_1h"] = grp["surface_pressure"].diff(1).fillna(0)
        grp["pressure_tendency_3h"] = grp["surface_pressure"].diff(3).fillna(0)
        # Dewpoint
        T = grp["temperature_2m"]; RH = grp["relative_humidity_2m"]
        alpha = (17.625 * T) / (243.04 + T) + np.log(RH / 100.0 + 1e-6)
        grp["dewpoint"] = (243.04 * alpha) / (17.625 - alpha)
        # Wind U/V
        wd_rad = np.deg2rad(grp["winddirection_10m"])
        grp["wind_u"] = (-grp["windspeed_10m"] * np.sin(wd_rad)).round(4)
        grp["wind_v"] = (-grp["windspeed_10m"] * np.cos(wd_rad)).round(4)
        # Lags
        grp["humidity_lag6h"] = grp["relative_humidity_2m"].shift(6).bfill()
        grp["precip_roll6h"]  = grp["precipitation"].rolling(6, min_periods=1).sum()
        
        # 3. Handle any residual NaNs (critical for live inference stability)
        grp = grp.ffill().bfill().fillna(0)
        
        processed_parts.append(grp)
    
    df = pd.concat(processed_parts)
    
    # 2. Spatial Lags
    seed_cols = ["surface_pressure", "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_tendency_1h"]
    seed_df = df[df["district"] == seed_district].set_index("timestamp")[seed_cols].sort_index()
    
    final_parts = []
    for district, grp in df.groupby("district"):
        grp = grp.set_index("timestamp").copy()
        lag = lag_map.get(district, 0)
        for col in seed_cols:
            lagged = seed_df[col].shift(lag, freq="h")
            grp[f"seed_{col}_lag{lag}h"] = lagged.reindex(grp.index).ffill().bfill()
        final_parts.append(grp.reset_index())
    
    df = pd.concat(final_parts)
    return df[feature_cols + ["district"]]

def main():
    parser = argparse.ArgumentParser(description="WeatherX Live Inference Utility")
    # Manual sensor overrides for Chennai
    parser.add_argument("--temp", "--temperature", type=float, help="Latest Temperature (°C)")
    parser.add_argument("--hum", "--humidity",   type=float, help="Latest Humidity (%)")
    parser.add_argument("--pres", "--pressure",  type=float, help="Latest Pressure (hPa)")
    parser.add_argument("--rain", "--precipitation", type=float, help="Latest Precipitation (mm)")
    parser.add_argument("--wspeed", "--windspeed", type=float, help="Latest Wind Speed (km/h)")
    parser.add_argument("--wdir", "--winddirection",  type=float, help="Latest Wind Direction (deg)")
    parser.add_argument("--gtemp", "--ground", type=float, help="Latest Ground Temperature (°C) [DS18B20]")
    
    parser.add_argument("--data-dir",   default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir",     default="models/saved_models")
    args = parser.parse_args()

    # 1. Load Inferencer & Meta
    try:
        # Check if the model exists before even importing the inferencer
        model_path = os.path.join(args.model_dir, "weatherx_multidistrict_lstm.h5")
        if not os.path.exists(model_path):
            print("\n[!] WARNING: The training process has not yet finished the first epoch.")
            print("    The forecasting engine will be ready as soon as the first checkpoint is saved.")
            print("    Please check again in ~1-2 hours.")
            return
            
        inferencer = WeatherXInferencer(processed_dir=args.processed_dir)
    except Exception as e:
        print(f"Error initializing inferencer: {e}")
        return

    meta = inferencer.meta
    districts = list(meta["districts"].keys())
    feature_cols = meta["feature_cols"]

    print(f"Ingesting latest 24h history for {len(districts)} districts...")
    
    # 2. Load latest 30 hours from CSVs (to ensure we have enough for 24h window + 6h lags)
    history_dfs = []
    for dist in districts:
        fpath = os.path.join(args.data_dir, f"{dist}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath).tail(48) # Get last 48h to be safe
            df["district"] = dist
            history_dfs.append(df)
    
    full_history = pd.concat(history_dfs, ignore_index=True)
    full_history["timestamp"] = pd.to_datetime(full_history["timestamp"], format='mixed', dayfirst=False)
    
    # 3. Prepare Overrides and Engineering
    overrides = {
        "temp": args.temp,
        "hum":  args.hum,
        "pres": args.pres,
        "rain": args.rain,
        "wspeed": args.wspeed,
        "wdir": args.wdir,
        "gtemp": args.gtemp
    }
    
    # 3.1 Ground-to-Air Micro-Climate Logic (DS18B20 Simulation)
    if args.gtemp is not None and args.temp is not None:
        delta_t = args.gtemp - args.temp
        if delta_t > 0:
            print(f"  [Micro-Climate] Ground > Air (+{delta_t:.1f}°C). Applying heatwave boost.")
            # Note: engineer_single_window will apply this shift globally if we update overrides
            overrides["temp"] += (delta_t * 0.2)
            
    # 4. Feature Engineering
    lag_map = {d: m["lag_hours"] for d, m in meta["districts"].items()}
    engineered_df = engineer_single_window(
        full_history, SEED_DISTRICT, lag_map, feature_cols, districts, 
        overrides=overrides
    ) 
    
    # 5. Scaling and Window Slicing
    # We need to extract the LAST 24 hours for each district and stack them
    X_list = []
    valid_names = []
    for dist in districts:
        dist_data = engineered_df[engineered_df["district"] == dist]
        
        # Extract the LAST 24 hours
        window_df = dist_data[feature_cols].tail(24)
        
        # Consistent NaN handling: match preprocess.py (filling with median)
        fill_values = {c: meta["feature_medians"].get(c, 0.0) for c in feature_cols}
        window = window_df.fillna(value=fill_values).ffill().bfill().values
        
        if window.shape[0] < 24:
             print(f"  [!] Skipping {dist}: insufficient history ({window.shape[0]}h < 24h)")
             continue

        # Scale
        if dist == SEED_DISTRICT:
            try:
                t_idx = feature_cols.index("seed_temperature_2m_lag0h")
                h_idx = feature_cols.index("seed_relative_humidity_2m_lag0h")
                print(f"DIAG 3 — {dist} Input RAW (Last 3h):")
                for k in range(-3, 0):
                    print(f"  [{k}] T:{window[k, t_idx]:.2f}, H:{window[k, h_idx]:.2f}")
            except: pass

        window_scaled = inferencer.feature_scaler.transform(window)
        X_list.append(window_scaled)
        valid_names.append(dist)
    
    if len(X_list) == 0:
        print("Error: No districts have sufficient historical data for inference.")
        return

    # 6. Prepare Batch (num_districts, 24, num_features)
    X_batch = np.array(X_list, dtype=np.float32)
    
    # DEBUG: Check if all inputs are the same
    print(f"DEBUG: X_batch shape: {X_batch.shape}")
    for i in range(min(5, len(valid_names))):
        print(f"DEBUG: {valid_names[i]} - sample feature mean: {X_batch[i].mean():.4f}, std: {X_batch[i].std():.4f}")
    
    if np.allclose(X_batch[0], X_batch[1]) if len(X_batch) > 1 else False:
        print("DEBUG ALERT: First two districts have IDENTICAL scaled features!")
    
    # 7. Predict
    print(f"\nRunning forecasting engine for {len(valid_names)} districts...")
    pred_descaled = inferencer.predict(
        X_batch,
        district_names=valid_names,
        sensor_temp=args.temp,
        sensor_hum=args.hum,
    )
    
    # AntiGravity Diagnostic (V5.2)
    for idx, dist in enumerate(valid_names):
        if dist in [SEED_DISTRICT, "Madurai"]:
            print(f"  [DIAG 4] {dist:12} | Raw Pred (1h): {pred_descaled[idx, 0]}")
    
    # Clamp to physical bounds — prevents impossible values during early training
    target_cols = inferencer.target_cols
    bounds = {
         "temperature_2m":        (-10, 55),
        "relative_humidity_2m":  (0, 100),
        "precipitation":         (0, 300),
    }
    for i, col in enumerate(target_cols):
        if col in bounds:
            lo, hi = bounds[col]
            pred_descaled[:, :, i] = np.clip(pred_descaled[:, :, i], lo, hi)
    
    report = inferencer.format_predictions(pred_descaled, valid_names)
    
    # 8. Output Report (Text & JSON)
    report_file = os.path.join(args.model_dir, "latest_forecast.txt")
    json_file   = os.path.join("dashboard", "latest_forecast.json")
    
    # --- Part A: Write Human Report ---
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write(f" WEATHERX SYSTEM — FULL 6H ALL-DISTRICT FORECAST\n")
        f.write(f" Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" Seed Sensor Station: {SEED_DISTRICT}\n")
        f.write("="*60 + "\n\n")
        
        for dist in sorted(valid_names):
            if dist in report:
                f.write(f">>> {dist}:\n")
                for h, data in enumerate(report[dist]):
                    # format_predictions already handles the units, we just write the rows
                    f.write(f"  [+{h+1}h] T:{data['temperature_2m']}°C, H:{data['relative_humidity_2m']}%, P:{data['precipitation']}mm\n")
                f.write("\n")

    # --- Part B: Export Dashboard JSON ---
    import json
    json_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "seed_district": SEED_DISTRICT,
        "districts": {}
    }
    
    for idx, dist in enumerate(valid_names):
        json_data["districts"][dist] = {
            "current": {
                "temp": float(pred_descaled[idx, 0, 1]),
                "hum":  float(pred_descaled[idx, 0, 2]),
                "rain": float(pred_descaled[idx, 0, 0])
            },
            "forecast": [
                {
                    "hour": i+1, 
                    "temp": float(pred_descaled[idx, i, 1]), 
                    "hum": float(pred_descaled[idx, i, 2]),
                    "rain": float(pred_descaled[idx, i, 0])
                } 
                for i in range(pred_descaled.shape[1])
            ]
        }
    
    # Ensure dashboard dir exists
    os.makedirs("dashboard", exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    
    print("\n" + "="*45)
    print(f" LIVE FORECAST FROM {SEED_DISTRICT} SENSORS")
    print(f" AI Intelligence exported to dashboard/latest_forecast.json")
    print("="*45)
    print(f"Full and final 39-district report saved to: \n  >> {report_file}")
    
    # Showcase for local and key regional districts
    display_list = [SEED_DISTRICT, "Tiruppur", "Erode", "Salem", "Chennai", "Madurai"]
    for dist in display_list:
        if dist in report:
            data_next = report[dist][0] # Next 1h
            print(f"  - {dist:15}: Next 1h -> T:{data_next['temperature_2m']}°C, H:{data_next['relative_humidity_2m']}%")
    
    print("\nCheck 'latest_forecast.txt' for the complete 6-hour breakdown.")

if __name__ == "__main__":
    main()
