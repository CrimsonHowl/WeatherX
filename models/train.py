"""
WeatherX — train.py
═══════════════════════════════════════════════════════════════════════════════
LSTM Training Script  |  V4
Fixes applied vs previous version:
  • weighted_mse loss from model.py — prevents precipitation-zero collapse
  • EarlyStopping monitors val_mae instead of val_loss — cannot be gamed
  • MinEpochEarlyStopping guarantees minimum 30 epochs before exit logic fires
  • ReduceLROnPlateau halves LR on plateau instead of quitting early
  • Default epochs increased 50 → 150
  • Per-target real-unit MAE printed at end of training
  • Log-scale training plot for better convergence visibility
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from models.model import build_weatherx_lstm, weighted_mse


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK — prevents early exit before model has had time to learn
# ─────────────────────────────────────────────────────────────────────────────

class MinEpochEarlyStopping(EarlyStopping):
    """
    EarlyStopping that cannot fire before `min_epoch`.

    The standard EarlyStopping will quit at epoch ~12 if val_loss collapses
    fast (e.g. the model learns to predict near-zero for everything and MSE
    looks artificially perfect).  This subclass silently skips patience
    tracking until the minimum epoch gate is passed.
    """
    def __init__(self, min_epoch: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epoch:
            # Gate not yet passed — skip EarlyStopping logic entirely
            if self.verbose > 0 and epoch == 0:
                print(
                    f"\nMinEpochEarlyStopping: patience tracking will begin "
                    f"at epoch {self.min_epoch}."
                )
            return
        super().on_epoch_end(epoch, logs)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss plot (log scale) ──
    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("WeatherX LSTM — Weighted MSE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted MSE (log scale)")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)

    # ── MAE plot (linear scale) ──
    ax2.plot(history.history["mae"],     label="Train MAE")
    ax2.plot(history.history["val_mae"], label="Val MAE")
    ax2.set_title("WeatherX LSTM — MAE (scaled units)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Training plot saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class WeatherXDataGenerator(tf.keras.utils.Sequence):
    """
    On-demand data generator that reads batches from disk to avoid
    memory exhaustion on Windows.
    """

    def __init__(
        self,
        processed_dir: str,
        split: str      = "train",
        batch_size: int = 64,
        shuffle: bool   = True,
    ):
        self.processed_dir = processed_dir
        self.split         = split
        self.batch_size    = batch_size
        self.shuffle       = shuffle

        meta_path = os.path.join(processed_dir, "preprocessing_meta.pkl")
        meta      = joblib.load(meta_path)

        self.district_info = []
        for dist, info in meta["districts"].items():
            x_path = os.path.join(
                processed_dir, "by_district", dist, f"X_{split}.npy"
            )
            y_path = os.path.join(
                processed_dir, "by_district", dist, f"y_{split}.npy"
            )
            count = info[f"n_{split}"]
            if count > 0 and os.path.exists(x_path) and os.path.exists(y_path):
                self.district_info.append(
                    {
                        "name":   dist,
                        "x_path": x_path,
                        "y_path": y_path,
                        "len":    count,
                    }
                )

        self.total_samples = sum(d["len"] for d in self.district_info)
        self.indices       = np.arange(self.total_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.total_samples / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Map global indices → (district, local_index)
        data_to_fetch = []
        for idx in batch_indices:
            curr = 0
            for d in self.district_info:
                if curr <= idx < curr + d["len"]:
                    data_to_fetch.append((d, idx - curr))
                    break
                curr += d["len"]

        # Sort by district to minimise file opens
        data_to_fetch.sort(key=lambda x: x[0]["name"])

        X_batch, y_batch             = [], []
        current_dist                 = None
        current_X = current_y = None

        for d, rel_idx in data_to_fetch:
            if d["name"] != current_dist:
                current_dist = d["name"]
                current_X    = np.load(d["x_path"], mmap_mode="r")
                current_y    = np.load(d["y_path"], mmap_mode="r")
            X_batch.append(current_X[rel_idx])
            y_batch.append(current_y[rel_idx])

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WeatherX LSTM Training Script")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir",     default="models/saved_models")
    parser.add_argument("--epochs",        type=int, default=150)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--min-epoch",     type=int, default=30,
                        help="Minimum epochs before EarlyStopping can fire")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    tf.keras.backend.clear_session()

    # ── 1. Data generators ───────────────────────────────────────────────────
    print(f"Initialising data generators from: {args.processed_dir}")
    train_gen = WeatherXDataGenerator(
        args.processed_dir,
        split="train",
        batch_size=args.batch_size,
    )
    val_gen = WeatherXDataGenerator(
        args.processed_dir,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
    )

    X_sample, y_sample = train_gen[0]
    seq_len      = X_sample.shape[1]
    num_features = X_sample.shape[2]
    horizon      = y_sample.shape[1]
    num_targets  = y_sample.shape[2]

    print(f"Train samples : {train_gen.total_samples:,}")
    print(f"Val samples   : {val_gen.total_samples:,}")
    print(f"Input shape   : ({seq_len}, {num_features})")
    print(f"Output shape  : ({horizon}, {num_targets})")

    # ── 2. Build model ───────────────────────────────────────────────────────
    model = build_weatherx_lstm(seq_len, num_features, horizon, num_targets)
    model.summary()

    # ── 3. Callbacks ─────────────────────────────────────────────────────────
    model_path = os.path.join(
        args.model_dir, "weatherx_multidistrict_lstm.h5"
    )

    callbacks = [
        # ── Gate: cannot fire before min_epoch ──────────────────────────────
        # ── Monitors val_mae (not val_loss) so near-zero precipitation
        #    cannot make the model look artificially converged ────────────────
        MinEpochEarlyStopping(
            min_epoch=args.min_epoch,
            monitor="val_mae",
            patience=20,
            min_delta=0.0001,
            restore_best_weights=True,
            verbose=1,
        ),
        # ── Save only the best checkpoint by val_mae ─────────────────────────
        ModelCheckpoint(
            model_path,
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
        # ── Halve LR when val_mae stalls — better than quitting ──────────────
        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Starting training with weighted MSE loss")
    print("  Target weights → Precip: 1.0 | Temp: 5.0 | Humidity: 3.0")
    print(f"  EarlyStopping gate: epoch {args.min_epoch}  |  patience: 20")
    print(f"  Max epochs: {args.epochs}")
    print("═" * 60 + "\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── 5. Per-target real-unit MAE report ───────────────────────────────────
    epochs_trained = len(history.history["loss"])
    final_mae      = history.history["mae"][-1]
    final_val_mae  = history.history["val_mae"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print("\n" + "═" * 60)
    print("  Training Complete")
    print("═" * 60)
    print(f"  Epochs trained : {epochs_trained}")
    print(f"  Final train MAE: {final_mae:.6f}  (scaled)")
    print(f"  Final val MAE  : {final_val_mae:.6f}  (scaled)")
    print(f"  Final val loss : {final_val_loss:.6f}")

    # Unscale MAE for human-readable interpretation
    try:
        meta          = joblib.load(
            os.path.join(args.processed_dir, "preprocessing_meta.pkl")
        )
        target_scaler = joblib.load("models/scalers/target_scaler.pkl")
        print("\n  Approx real-unit val MAE per target:")
        for i, col in enumerate(meta["target_cols"]):
            approx = final_val_mae / target_scaler.scale_[i]
            print(f"    {col:35}: ±{approx:.2f}")
    except Exception as e:
        print(f"  (Could not compute real-unit MAE: {e})")

    print("═" * 60 + "\n")

    # ── 6. Save plot ─────────────────────────────────────────────────────────
    plot_path = os.path.join(args.model_dir, "training_loss.png")
    plot_training_history(history, plot_path)
    print(f"Model saved to : {model_path}")


if __name__ == "__main__":
    main()