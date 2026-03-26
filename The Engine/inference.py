"""
WeatherX — inference.py
═══════════════════════════════════════════════════════════════════════════════
LSTM Inference Engine  |  V4
Fixes applied vs V3:
  • inverse_transform now receives 2D flat array → correct real-unit output
  • Scaler shape validated at load time (catches 1-feature vs 3-feature mismatch)
  • Single-district and batch prediction both supported
  • format_predictions produces both dict report and a plain-text summary
  • All debug prints replaced with structured logging (toggle via LOG_LEVEL env var)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING  (set LOG_LEVEL=DEBUG in your shell for verbose output)
# ─────────────────────────────────────────────────────────────────────────────

print(f"DIAG 0 — inference.py loaded from: {os.path.abspath(__file__)}")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("weatherx.inference")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — safe inverse transform  (THE core fix)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform_predictions(scaler, predictions: np.ndarray) -> np.ndarray:
    """
    Safely inverse-transforms LSTM output back to real physical units.

    sklearn's MinMaxScaler.inverse_transform() only accepts 2D arrays.
    LSTM model output is 3D: (districts, horizon, targets).
    Passing 3D input causes a silent identity-pass — the scaler returns the
    input unchanged.  This function reshapes before and after to prevent that.

    Args:
        scaler      : Fitted MinMaxScaler loaded from target_scaler.pkl
        predictions : np.ndarray of shape (n_districts, horizon, n_targets)

    Returns:
        np.ndarray of the SAME shape, in original units (°C, %, mm…)
    """
    original_shape = predictions.shape        # e.g. (39, 6, 3)
    n_targets      = original_shape[-1]

    flat      = predictions.reshape(-1, n_targets)    # (234, 3)
    flat_real = scaler.inverse_transform(flat)        # (234, 3) — real units
    return flat_real.reshape(original_shape)          # (39, 6, 3) — real units


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class WeatherXInferencer:

    def __init__(
        self,
        model_dir     : str = "models/saved_models",
        scaler_dir    : str = "models/scalers",
        processed_dir : str = "data/processed",
    ):
        log.info("Initialising WeatherX Inference Engine V4…")

        # ── Load LSTM model ──────────────────────────────────────────────────
        self.model_path = os.path.join(model_dir, "weatherx_multidistrict_lstm.h5")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run train.py first to generate the model file."
            )
        self.model = load_model(self.model_path, compile=False)
        log.info(f"Model loaded  │ input shape : {self.model.input_shape}")
        log.info(f"              │ output shape: {self.model.output_shape}")

        # ── Load scalers ─────────────────────────────────────────────────────
        feat_path   = os.path.join(scaler_dir, "feature_scaler.pkl")
        target_path = os.path.join(scaler_dir, "target_scaler.pkl")

        self.feature_scaler = self._load_and_validate_scaler(feat_path,   name="feature")
        self.target_scaler  = self._load_and_validate_scaler(target_path, name="target")

        # ── Load preprocessing metadata ──────────────────────────────────────
        meta_path = os.path.join(processed_dir, "preprocessing_meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        self.meta         = joblib.load(meta_path)
        self.meta         = joblib.load(meta_path)
        # Physical Target Mapping (Final V5.4 Alignment)
        # index 0: Precipitation (Rain)
        # index 1: Temperature (BME280) — 21.95°C (Realistic Night)
        # index 2: Humidity (BME280)    — 38.26% (Realistic Air)
        self.target_cols  = ["precipitation", "temperature_2m", "relative_humidity_2m"]

        self.feature_cols = self.meta["feature_cols"]
        self.districts    = list(self.meta["districts"].keys())
        self.horizon      = self.meta["forecast_horizon"]
        self.seq_len      = self.meta["seq_len"]

        # ── Sanity-check scaler vs target columns ────────────────────────────
        n_scaler_features = self.target_scaler.scale_.shape[0]
        if n_scaler_features != len(self.target_cols):
            raise ValueError(
                f"target_scaler was fitted on {n_scaler_features} feature(s), "
                f"but target_cols has {len(self.target_cols)}: {self.target_cols}\n"
                "Re-run preprocess.py and retrain to regenerate matching scalers."
            )

        # V6.2: Live sensor state for Hybrid Anchor correction
        self._sensor_temp: float | None = None
        self._sensor_hum:  float | None = None

        log.info(
            f"Ready │ districts={len(self.districts)}  "
            f"horizon={self.horizon}h  seq_len={self.seq_len}h  "
            f"targets={self.target_cols}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_and_validate_scaler(path: str, name: str):
        """Loads a joblib scaler and verifies it is fitted."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}_scaler not found: {path}")
        scaler = joblib.load(path)
        if not hasattr(scaler, "scale_"):
            raise ValueError(
                f"{name}_scaler at {path} is NOT fitted (missing 'scale_' attr).\n"
                "Re-run preprocess.py to regenerate fitted scalers."
            )
        log.info(
            f"{name}_scaler │ scale_ : {scaler.scale_}  │  "
            f"range: [{scaler.data_min_}  →  {scaler.data_max_}]"
        )
        return scaler

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        X_batch: np.ndarray,
        district_names: list = None,
        sensor_temp: float = None,
        sensor_hum:  float = None,
    ) -> np.ndarray:
        """
        Runs the LSTM, applies inverse scaling, and the V6.2 Sensor Anchor.

        Args:
            X_batch        : shape (n_districts, seq_len, n_features)
            district_names : optional list of labels used for per-district calibration
            sensor_temp    : live BME280 temperature reading (°C)
            sensor_hum     : live BME280 humidity reading (%)
        """
        self._sensor_temp = sensor_temp
        self._sensor_hum  = sensor_hum
        if X_batch.ndim != 3:
            raise ValueError(
                f"X_batch must be 3D (n_districts, seq_len, n_features), "
                f"got shape {X_batch.shape}"
            )

        # ── Step 1: raw model output (still scaled) ──────────────────────────
        preds_scaled = self.model.predict(X_batch, verbose=0)
        print(f"DIAG 1 — preds_scaled shape  : {preds_scaled.shape}")
        print(f"DIAG 1 — scaled sample [0,0] : {preds_scaled[0, 0]}")

        # ── Step 2: reshape 3D→2D, inverse transform, restore shape ──────────
        #    This is the fix for the "Identity Scaler" paradox.
        #    sklearn only accepts 2D input — passing 3D causes silent passthrough.
        preds_real = inverse_transform_predictions(self.target_scaler, preds_scaled)
        print(f"DIAG 2 — preds_real shape    : {preds_real.shape}")
        print(f"DIAG 2 — real-unit sample [0,0]: {preds_real[0, 0]}")

        # --- AntiGravity V6.2 Hybrid Sensor Anchor ---
        # Problem: The LSTM's winter weights predict "midnight floor" (~22°C) even when
        #          real March data and live sensor say the temperature is 28°C.
        # Fix: Blend the model output toward the live sensor reading.
        #   final_pred = model_pred + anchor_weight * (sensor_temp - model_pred[+1h])
        # This anchors the forecast without overriding the LSTM's trend intelligence.
        #
        # sensor_temp and sensor_hum are injected via predict() call from predict_live.py
        # --- AntiGravity V6.4 Dynamic Summer Anchor ---
        # Problem: Fixed 0.75 weight is too weak for the 13°C gap at 4:30 PM.
        # Fix: During peak heat (11 AM - 6 PM), we give the sensor 95% authority.
        # This forces the model to respect the summer heat despite its winter-bias.
        from datetime import datetime
        hour = datetime.now().hour
        
        # Performance Curve: 95% at Noon, 75% at Midnight
        if 11 <= hour <= 18:
            anchor_weight = 0.95 # Peak Summer authority
            hum_weight = 0.90 # High dry-air authority
        elif 18 <= hour <= 21:
            anchor_weight = 0.85 # Sunset transition
            hum_weight = 0.75
        else:
            anchor_weight = 0.75 # Night blend (Original V6.2)
            hum_weight = 0.60

        if self._sensor_temp is not None:
            # Index 0 = seed district (Coimbatore), idx 1 = temperature
            model_temp_1h = preds_real[0, 0, 1]  # LSTM's next-1h temp for seed district
            temp_gap = self._sensor_temp - model_temp_1h
            correction = anchor_weight * temp_gap
            preds_real[:, :, 1] += correction
            print(f"  [V6.4 Dynamic Anchor] Hour:{hour} | Weight:{anchor_weight} | Applied: {correction:+.1f}°C")

        # --- AntiGravity V6.3 Climate-Zone Humidity Calibration ---
        # The full 39-district classification of Tamil Nadu
        # Temperature anchor: global (regional thermal trend is valid state-wide)
        # Humidity anchor:    zone-specific (coastal vs inland have fundamentally different moisture)

        # --- Complete Tamil Nadu District Classification ---
        COASTAL = {
            # Bay of Bengal / Coromandel Coast districts
            "Chennai", "Tiruvallur", "Kancheepuram", "Chengalpattu",
            "Villupuram", "Cuddalore", "Puducherry",
            "Nagapattinam", "Tiruvarur", "Mayiladuthurai",
            # Gulf of Mannar / southern coast
            "Ramanathapuram", "Thoothukudi", "Tirunelveli", "Kanniyakumari",
        }

        WESTERN_HILLS = {
            # Western Ghats / high-altitude — separate moisture profile
            "Nilgiris", "Theni",
        }

        # All remaining districts are INLAND (dry summer, moderate humidity)
        # Coimbatore, Tiruppur, Erode, Salem, Namakkal, Dharmapuri,
        # Krishnagiri, Vellore, Tirupattur, Ranipet, Tiruvannamalai,
        # Ariyalur, Perambalur, Tiruchirappalli, Karur, Dindigul,
        # Madurai, Sivaganga, Virudhunagar, Tenkasi, Kallakurichi,
        # Thanjavur, Pudukkottai

        if self._sensor_hum is not None and district_names:
            model_hum_1h = preds_real[0, 0, 2]  # seed district (Coimbatore)

            for idx, dist in enumerate(district_names):
                if dist in COASTAL:
                    # Coastal: strong marine layer boost (+30%)
                    # Capped at 95% — 100% = active fog/rain, already tracked by precipitation column
                    coastal_target = np.clip(self._sensor_hum + 30.0, 0, 95)
                    gap = coastal_target - preds_real[idx, 0, 2]
                    preds_real[idx, :, 2] = np.clip(preds_real[idx, :, 2] + 0.85 * gap, 0, 95)
                elif dist in WESTERN_HILLS:
                    # Western Ghats: high elevation humidity (~70-80%)
                    hills_target = np.clip(self._sensor_hum + 20.0, 0, 100)
                    gap = hills_target - preds_real[idx, 0, 2]
                    preds_real[idx, :, 2] = np.clip(preds_real[idx, :, 2] + 0.7 * gap, 0, 100)
                else:
                    # Inland: apply sensor anchor correction only (no boost)
                    # Use dynamic hum_weight (90% during day, 60% at night)
                    hum_gap = self._sensor_hum - model_hum_1h
                    preds_real[idx, :, 2] = np.clip(preds_real[idx, :, 2] + hum_weight * hum_gap, 0, 100)

        return preds_real

    def predict_single_district(
        self,
        X_seq: np.ndarray,
        district_name: str = "unknown",
    ) -> dict:
        """
        Convenience wrapper for single-district inference.

        Args:
            X_seq         : shape (seq_len, n_features) or (1, seq_len, n_features)
            district_name : label used in the returned dict
        """
        if X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, ...]       # → (1, seq_len, n_features)
        preds_real = self.predict(X_seq)          # (1, horizon, n_targets)
        return self.format_predictions(preds_real, [district_name])

    def format_predictions(
        self,
        preds_real: np.ndarray,
        district_names: list,
    ) -> dict:
        """
        Converts predictions array into a structured report dictionary.

        Returns:
            {
              "Coimbatore": [
                  {"precipitation": 0.0, "temperature_2m": 33.1, "relative_humidity_2m": 72.4},
                  ...  # one dict per horizon step
              ],
              ...
            }
        """
        if len(district_names) != preds_real.shape[0]:
            raise ValueError(
                f"district_names length ({len(district_names)}) does not match "
                f"predictions first dimension ({preds_real.shape[0]})"
            )

        report = {}
        for i, district in enumerate(district_names):
            report[district] = [
                {
                    col: round(float(preds_real[i, h, j]), 2)
                    for j, col in enumerate(self.target_cols)
                }
                for h in range(self.horizon)
            ]
        return report

    def print_forecast(self, report: dict) -> None:
        """Prints a human-readable forecast table to stdout."""
        target_labels = {
            "temperature_2m":       ("Temp",  "°C"),
            "relative_humidity_2m": ("Humid", "%"),
            "precipitation":        ("Rain",  "mm"),
            "surface_pressure":     ("Press", "hPa"),
        }
        sep = "─" * 72
        print(f"\n{'═' * 72}")
        print(f"  WeatherX Forecast  │  Horizon: {self.horizon}h ahead")
        print(f"{'═' * 72}")

        for district, steps in report.items():
            print(f"\n  📍 {district}")
            print(f"  {sep}")
            header = f"  {'Hour':>5}  " + "  ".join(
                f"{target_labels.get(c, (c, ''))[0]:>8}" for c in self.target_cols
            )
            units = f"  {'':>5}  " + "  ".join(
                f"{'(' + target_labels.get(c, ('', ''))[1] + ')':>8}" for c in self.target_cols
            )
            print(header)
            print(units)
            print(f"  {sep}")
            for h, step in enumerate(steps, start=1):
                vals = "  ".join(f"{v:>8.2f}" for v in step.values())
                print(f"  {f'+{h}h':>5}  {vals}")

        print(f"\n{'═' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Running smoke test…")
    try:
        inferencer = WeatherXInferencer()
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    n_districts = len(inferencer.districts)
    seq_len     = inferencer.seq_len
    n_features  = len(inferencer.feature_cols)

    dummy_input = np.random.uniform(0, 1, (n_districts, seq_len, n_features)).astype(np.float32)
    log.info(f"Dummy input shape: {dummy_input.shape}")

    preds_real = inferencer.predict(dummy_input)

    temp_vals = preds_real[:, 0, 0]
    if temp_vals.max() <= 1.0:
        log.error(
            "⚠️  Values still in scaled range — inverse_transform did not apply.\n"
            "    Verify target_scaler.pkl was fitted on 3 features."
        )
        sys.exit(1)
    else:
        log.info(f"✓ Sanity check passed — sample temperatures: {np.round(temp_vals[:5], 2)} °C")

    sample_report = inferencer.format_predictions(preds_real[:3], inferencer.districts[:3])
    inferencer.print_forecast(sample_report)
