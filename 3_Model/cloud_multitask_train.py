#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task cloud evolution forecasting (scriptified from model_definition_evaluation_MF_multitask_v2.ipynb)

Key features:
- CLI-configurable hyperparameters (batch, epochs, lr, loss, in/out window, stride, RNN type/units/layers, dropout, etc.)
- Feature cleanup: removes lwp_L00 / iwp_L00, uses tqc_L00 / tqi_L00
- Adds engineered profile features (max/height_max/sum for qc/qi/qr/qg/qs)
- Adds scalar ratios:
    - ratio_tqc_tqi = tqc_L00 / (tqi_L00 + eps)
    - rain_efficiency = tqc_L00 / (rain_gsp_rate_L00 + eps)
- Adds motion features computed from lon/lat (relative dx/dy, speed, heading sin/cos, accel, turn_rate)
- Loads data from Hugging Face dataset repo OR local directory (or both)
- Saves full outputs locally (and optionally uploads to HF if configured):
    - model, config, scalers, predictions, metrics, plots, history

Notes:
- This script intentionally DOES NOT embed any HF token. Use `--hf-token` or env var HF_TOKEN.
- Column names for lon/lat are configurable; defaults are "lon" and "lat".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# TensorFlow (Keras)
import tensorflow as tf
from tensorflow.keras import layers, models

# sklearn
from sklearn.preprocessing import StandardScaler
import joblib

# plotting
import matplotlib.pyplot as plt

# huggingface (optional)
try:
    from huggingface_hub import login as hf_login
    from huggingface_hub import list_repo_files, hf_hub_download
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# -------------------------
# Notebook-consistent names
# -------------------------
PROFILE_PREFIXES = ["qr_", "qc_", "qi_", "qs_", "qg_", "qv_", "roh_", "w_"]

# We will override SCALAR_FEATURES from the notebook by removing lwp/iwp and keeping remaining_lifetime for labels only.
SCALAR_FEATURES_BASE = [
    "cape_ml_L00", "cin_ml_L00",
    "rain_gsp_rate_L00",
    "tqc_L00", "tqi_L00",
    "area_m2",
    "remaining_lifetime",  # keep for labels only, excluded from X
]

# names/order of engineered time-series features (base set from the notebook)
TS_FEATURE_NAMES_BASE = [
    "cloud_base",
    "cloud_top",
    "cloud_thickness",
    "cloud_mass",
    "rain_mass",
    "mean_w",
    "max_w",
    "height_max_qc",
    "height_max_w",
    "center_of_mass",
    "max_qc",
    "std_w_in_cloud",
]


# -------------------------
# Utility helpers
# -------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def safe_div(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return numer / (denom + eps)


# -------------------------
# Data reading / listing
# -------------------------
def list_hf_csv_files(
    repo_id: str,
    repo_subdir: str,
    track_len_csv_in_repo: Optional[str],
    min_track_len: Optional[int],
    datashare: float,
    seed: int,
) -> List[str]:
    if not HF_AVAILABLE:
        raise RuntimeError("huggingface_hub is not available in this environment. Install it or use --data-source local.")
    files = list_repo_files(repo_id, repo_type="dataset")
    csv_files = [f for f in files if f.startswith(repo_subdir.rstrip("/") + "/") and f.endswith(".csv")]

    if track_len_csv_in_repo and min_track_len:
        local_track_len = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=track_len_csv_in_repo)
        track_len_df = pd.read_csv(local_track_len)
        if "track_len" not in track_len_df.columns or "filename" not in track_len_df.columns:
            raise ValueError(f"track_len_csv must have columns ['filename','track_len']. Found: {track_len_df.columns.tolist()}")
        track_len_df = track_len_df[track_len_df["track_len"] >= int(min_track_len)]
        allowed = set(track_len_df["filename"].astype(str).tolist())
        csv_files = [f for f in csv_files if f.split("/")[-1] in allowed]

    rnd = random.Random(seed)
    rnd.shuffle(csv_files)

    n = len(csv_files)
    n_take = int(datashare * n)
    csv_files = csv_files[:n_take]
    return csv_files


def list_local_csv_files(
    local_data_path: Path,
    data_glob: str,
    track_len_csv_local: Optional[Path],
    min_track_len: Optional[int],
    datashare: float,
    seed: int,
) -> List[Path]:
    csv_files = sorted(local_data_path.glob(data_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{local_data_path}' matching glob '{data_glob}'")

    if track_len_csv_local and min_track_len:
        track_len_df = pd.read_csv(track_len_csv_local)
        if "track_len" not in track_len_df.columns or "filename" not in track_len_df.columns:
            raise ValueError(f"track_len_csv must have columns ['filename','track_len']. Found: {track_len_df.columns.tolist()}")
        track_len_df = track_len_df[track_len_df["track_len"] >= int(min_track_len)]
        allowed = set(track_len_df["filename"].astype(str).tolist())
        csv_files = [p for p in csv_files if p.name in allowed]

    rnd = random.Random(seed)
    rnd.shuffle(csv_files)

    n = len(csv_files)
    n_take = int(datashare * n)
    csv_files = csv_files[:n_take]
    return csv_files


def split_files(files: Sequence, seed: int, train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[List, List, List]:
    rnd = random.Random(seed)
    files = list(files)
    rnd.shuffle(files)
    n = len(files)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


# -------------------------
# Feature extraction
# -------------------------
def compute_remaining_lifetime(df: pd.DataFrame, timestep_minutes: int = 5) -> List[float]:
    n = len(df)
    return [(n - i - 1) * timestep_minutes for i in range(n)]


def extract_profile(df: pd.DataFrame, prefix: str, n_levels: int = 50) -> np.ndarray:
    """Extract vertical profiles with fixed n_levels; missing columns are filled with 0."""
    data = np.zeros((len(df), n_levels), dtype="float32")
    for i in range(n_levels):
        col = f"{prefix}L{i:02d}"
        if col in df.columns:
            data[:, i] = df[col].values
    return data


def _z_levels() -> np.ndarray:
    # hard-coded z-levels from the notebook (Z=50, index 0 = top layer)
    return np.array([
        3.136780e+04, 2.736595e+04, 2.492369e+04, 2.294698e+04, 2.125334e+04,
        1.975951e+04, 1.841803e+04, 1.719845e+04, 1.607970e+04, 1.504641e+04,
        1.408694e+04, 1.319217e+04, 1.235481e+04, 1.156892e+04, 1.082956e+04,
        1.013258e+04, 9.474455e+03, 8.852140e+03, 8.263009e+03, 7.704765e+03,
        7.175387e+03, 6.673090e+03, 6.196285e+03, 5.743555e+03, 5.313629e+03,
        4.905366e+03, 4.517735e+03, 4.149806e+03, 3.800737e+03, 3.469765e+03,
        3.156199e+03, 2.859414e+03, 2.578843e+03, 2.313976e+03, 2.064356e+03,
        1.829575e+03, 1.609273e+03, 1.403137e+03, 1.210904e+03, 1.032357e+03,
        8.674825e+02, 7.148409e+02, 5.740922e+02, 4.448988e+02, 3.269299e+02,
        2.198620e+02, 1.233794e+02, 3.718376e+01, -3.901340e+01, -1.055560e+02,
    ], dtype="float32")


def extract_ts_features_from_profiles(
    profiles: np.ndarray,
    cloud_threshold: float = 1e-12,
    add_hydrometeor_stats: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    profiles: (T, Z, F) with F=len(PROFILE_PREFIXES), Z must be 50
    Returns:
      ts_features: (T, D)
      ts_feature_names: list of names in order
    """
    if profiles.ndim != 3:
        raise ValueError(f"profiles must be 3D (T,Z,F). Got shape {profiles.shape}")

    z = _z_levels()
    if profiles.shape[1] != len(z):
        raise ValueError(f"Expected Z={len(z)} levels, got {profiles.shape[1]}")

    # indices consistent with PROFILE_PREFIXES
    idx = {p: i for i, p in enumerate(PROFILE_PREFIXES)}
    # required base profiles (from the notebook's logic)
    qc = profiles[:, :, idx["qc_"]]
    qi = profiles[:, :, idx["qi_"]]
    qr = profiles[:, :, idx["qr_"]]
    w  = profiles[:, :, idx["w_"]]

    ts_features = []
    for t in range(profiles.shape[0]):
        qc_t = qc[t]
        qi_t = qi[t]
        qr_t = qr[t]
        w_t  = w[t]

        cloud_mask = (qc_t + qi_t) > cloud_threshold

        if not np.any(cloud_mask):
            # keep shape stable with zeros
            ts_features.append([0.0] * len(TS_FEATURE_NAMES_BASE))
            continue

        z_cloud = z[cloud_mask]

        cloud_base = float(np.min(z_cloud))
        cloud_top  = float(np.max(z_cloud))
        cloud_thickness = cloud_top - cloud_base

        cloud_mass = float(np.sum(qc_t + qi_t))
        rain_mass  = float(np.sum(qr_t))

        w_in_cloud = w_t[cloud_mask]
        mean_w = float(np.mean(w_in_cloud))
        max_w  = float(np.max(w_in_cloud))

        height_max_qc = float(z[int(np.argmax(qc_t))])
        height_max_w  = float(z[int(np.argmax(w_t))])

        weights = qc_t + qi_t
        center_of_mass = float(np.sum(z * weights) / (cloud_mass + 1e-12))

        base = [
            cloud_base,
            cloud_top,
            cloud_thickness,
            cloud_mass,
            rain_mass,
            mean_w,
            max_w,
            height_max_qc,
            height_max_w,
            center_of_mass,
            float(np.max(qc_t)),
            float(np.std(w_in_cloud)),
        ]
        ts_features.append(base)

    ts_arr = np.array(ts_features, dtype="float32")
    names = list(TS_FEATURE_NAMES_BASE)

    if add_hydrometeor_stats:
        # add max/height_max/sum for qc, qi, qr, qg, qs
        hydros = ["qc_", "qi_", "qr_", "qg_", "qs_"]
        extra_feats = []
        extra_names = []
        z = _z_levels()

        for hydro in hydros:
            h = profiles[:, :, idx[hydro]]  # (T,Z)
            max_h = np.max(h, axis=1)
            argmax_h = np.argmax(h, axis=1)
            height_max_h = z[argmax_h]
            sum_h = np.sum(h, axis=1)

            extra_feats.append(max_h.astype("float32"))
            extra_feats.append(height_max_h.astype("float32"))
            extra_feats.append(sum_h.astype("float32"))

            base_name = hydro.replace("_", "")
            extra_names.extend([f"max_{base_name}", f"height_max_{base_name}", f"sum_{base_name}"])

        extra = np.stack(extra_feats, axis=1)  # (T, 3*len(hydros))
        ts_arr = np.concatenate([ts_arr, extra], axis=1)
        names.extend(extra_names)

    return ts_arr, names


def add_motion_features(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    step_seconds: int,
) -> pd.DataFrame:
    """
    Add relative motion features computed from lon/lat:
      dx, dy (meters per step), speed (m/s), heading_sin/cos, acceleration, turn_rate (rad/s)
    If lon/lat are missing, this function leaves df unchanged.
    """
    if lon_col not in df.columns or lat_col not in df.columns:
        return df

    lon = df[lon_col].astype("float64").to_numpy()
    lat = df[lat_col].astype("float64").to_numpy()

    # Equirectangular approximation for small distances:
    # dx = R * cos(lat0) * dlon, dy = R * dlat
    R = 6371000.0  # meters
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    dlon = np.diff(lon_rad, prepend=lon_rad[0])
    dlat = np.diff(lat_rad, prepend=lat_rad[0])

    lat0 = lat_rad
    dx = R * np.cos(lat0) * dlon
    dy = R * dlat

    speed = np.sqrt(dx * dx + dy * dy) / max(step_seconds, 1)

    heading = np.arctan2(dy, dx)  # rad
    heading_sin = np.sin(heading)
    heading_cos = np.cos(heading)

    accel = np.diff(speed, prepend=speed[0]) / max(step_seconds, 1)
    turn_rate = np.diff(heading, prepend=heading[0]) / max(step_seconds, 1)

    out = df.copy()
    out["dx_m"] = dx.astype("float32")
    out["dy_m"] = dy.astype("float32")
    out["speed_mps"] = speed.astype("float32")
    out["heading_sin"] = heading_sin.astype("float32")
    out["heading_cos"] = heading_cos.astype("float32")
    out["accel_mps2"] = accel.astype("float32")
    out["turn_rate_rps"] = turn_rate.astype("float32")
    return out


def preprocess_cloud(
    df: pd.DataFrame,
    scalar_features: List[str],
    step_seconds: int,
    lon_col: str,
    lat_col: str,
    add_motion: bool,
    add_scalar_ratios: bool,
    add_hydrometeor_stats: bool,
    n_profile_levels: int = 50,
) -> Dict[str, np.ndarray]:
    df = df.sort_values("time") if "time" in df.columns else df

    # interpolate cin/cape as in notebook
    if "cin_ml_L00" in df.columns:
        df["cin_ml_L00"] = df["cin_ml_L00"].interpolate(method="linear").bfill()
    if "cape_ml_L00" in df.columns:
        df["cape_ml_L00"] = df["cape_ml_L00"].interpolate(method="linear").bfill()

    # remaining_lifetime as in notebook
    if "remaining_lifetime" not in df.columns:
        df["remaining_lifetime"] = compute_remaining_lifetime(df)

    if add_motion:
        df = add_motion_features(df, lon_col=lon_col, lat_col=lat_col, step_seconds=step_seconds)

    # profiles (T,Z,F)
    profile_features = []
    for prefix in PROFILE_PREFIXES:
        prof = extract_profile(df, prefix, n_levels=n_profile_levels)
        profile_features.append(prof)
    profiles = np.stack(profile_features, axis=-1).astype("float32")

    # engineered ts_features
    ts_features, ts_names = extract_ts_features_from_profiles(
        profiles, add_hydrometeor_stats=add_hydrometeor_stats
    )

    # scalars (T, S)
    missing = [c for c in scalar_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required scalar columns: {missing}")

    scalars = df[scalar_features].astype("float32").to_numpy()

    # add scalar ratios (append new columns after scalars)
    extra_scalar_names = []
    extra_scalar_cols = []

    if add_scalar_ratios:
        # ratio_tqc_tqi
        tqc = df["tqc_L00"].astype("float32").to_numpy()
        tqi = df["tqi_L00"].astype("float32").to_numpy()
        ratio = safe_div(tqc, tqi, eps=1e-12).astype("float32")
        extra_scalar_names.append("ratio_tqc_tqi")
        extra_scalar_cols.append(ratio)

        # rain_efficiency = total qc / rainrate (as requested)
        rainrate = df["rain_gsp_rate_L00"].astype("float32").to_numpy()
        eff = safe_div(tqc, rainrate, eps=1e-12).astype("float32")
        extra_scalar_names.append("rain_efficiency_tqc_over_rainrate")
        extra_scalar_cols.append(eff)

    # append motion scalars if present
    motion_scalar_names = []
    motion_cols = []
    for c in ["dx_m", "dy_m", "speed_mps", "heading_sin", "heading_cos", "accel_mps2", "turn_rate_rps"]:
        if c in df.columns:
            motion_scalar_names.append(c)
            motion_cols.append(df[c].astype("float32").to_numpy())

    if extra_scalar_cols or motion_cols:
        appended = []
        appended_names = []
        if extra_scalar_cols:
            appended.append(np.stack(extra_scalar_cols, axis=1))
            appended_names.extend(extra_scalar_names)
        if motion_cols:
            appended.append(np.stack(motion_cols, axis=1))
            appended_names.extend(motion_scalar_names)

        scalars = np.concatenate([scalars] + appended, axis=1).astype("float32")
        scalar_features = scalar_features + appended_names

    return {
        "profiles": profiles,            # (T, Z, F)
        "ts_features": ts_features,      # (T, D_ts)
        "ts_names": np.array(ts_names, dtype=object),
        "scalars": scalars,              # (T, S_total)
        "scalar_names": np.array(scalar_features, dtype=object),
    }


# -------------------------
# Windowing (multi-task)
# -------------------------
def _pick_feature_name(available: Sequence[str], candidates: Sequence[str], label: str) -> str:
    for c in candidates:
        if c in available:
            print(f"[window] Using {label} feature: '{c}'")
            return c
    raise ValueError(f"Could not find a valid {label} feature. Tried: {candidates}. Available: {available}")


def build_windows_multitask(
    samples: List[Dict[str, np.ndarray]],
    n_input_steps: int,
    n_output_steps: int,
    stride: int,
    targets: List[str],
    rul_feature_candidates: Sequence[str] = ("remaining_lifetime", "remaining_time", "rul", "ttl_remaining", "time_to_end"),
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Create (X, Y_dict) windows for multi-task forecasting.

    X per step = [scalar inputs (excluding RUL) + engineered ts_features]
    Y per task = (n_output_steps,) sequence

    Returns:
      X: (N, n_input_steps, n_features_total)
      Y: dict task -> (N, n_output_steps)
      meta: dict with feature names lists: X_feature_names, scalar_input_features, ts_feature_names
    """
    if not samples:
        raise ValueError("No samples provided")

    # infer feature names from first sample
    scalar_names = samples[0]["scalar_names"].tolist()
    ts_names = samples[0]["ts_names"].tolist()

    # detect RUL feature in scalars (for labels only)
    rul_name = _pick_feature_name(scalar_names, rul_feature_candidates, "remaining lifetime (RUL)")

    scalar_input_features = [f for f in scalar_names if f != rul_name]
    scalar_input_idxs = [scalar_names.index(f) for f in scalar_input_features]

    # indices for some potential targets that live in ts_features
    ts_name_to_idx = {n: i for i, n in enumerate(ts_names)}
    scalar_name_to_idx = {n: i for i, n in enumerate(scalar_names)}

    X_list = []
    Y = {t: [] for t in targets}

    for s in samples:
        scal = s["scalars"]               # (T,S)
        tsf  = s["ts_features"]           # (T,D)

        T = scal.shape[0]
        # Require enough length for one window
        total_needed = n_input_steps + n_output_steps
        if T < total_needed:
            continue

        # Build X features (exclude RUL)
        X_full = np.concatenate([scal[:, scalar_input_idxs], tsf], axis=1).astype("float32")
        # Precompute target series
        target_series = {}
        for tname in targets:
            if tname in scalar_name_to_idx:
                target_series[tname] = scal[:, scalar_name_to_idx[tname]].astype("float32")
            elif tname in ts_name_to_idx:
                target_series[tname] = tsf[:, ts_name_to_idx[tname]].astype("float32")
            elif tname == "rul":
                # allow alias: task called "rul" uses detected rul_name in scalars
                target_series[tname] = scal[:, scalar_name_to_idx[rul_name]].astype("float32")
            else:
                raise KeyError(
                    f"Target '{tname}' not found in scalars or ts_features. "
                    f"Known scalars: {list(scalar_name_to_idx)[:10]}... , ts: {list(ts_name_to_idx)[:10]}..."
                )

        # sliding windows with stride
        for start in range(0, T - total_needed + 1, stride):
            x_win = X_full[start:start + n_input_steps]
            y_start = start + n_input_steps
            y_end = y_start + n_output_steps

            X_list.append(x_win)

            for tname in targets:
                y_win = target_series[tname][y_start:y_end]
                Y[tname].append(y_win)

    if not X_list:
        raise ValueError("No windows created. Check cutoff lengths, input/output steps, stride, and filtering.")

    X = np.stack(X_list, axis=0).astype("float32")
    Y_out = {k: np.stack(v, axis=0).astype("float32") for k, v in Y.items()}

    meta = {
        "scalar_input_features": scalar_input_features,
        "ts_feature_names": ts_names,
        "X_feature_names": scalar_input_features + ts_names,
        "rul_feature_name_in_scalars": rul_name,
    }
    return X, Y_out, meta


# -------------------------
# Scaling
# -------------------------
@dataclass
class Scalers:
    x_scaler: StandardScaler
    y_scalers: Dict[str, StandardScaler]


def fit_transform_scalers(
    X_train: np.ndarray,
    Y_train: Dict[str, np.ndarray],
    scale_y: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Scalers]:
    # X scaler on flattened features
    x_scaler = StandardScaler()
    Xf = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = x_scaler.fit_transform(Xf).reshape(X_train.shape).astype("float32")

    y_scalers: Dict[str, StandardScaler] = {}
    Y_train_scaled: Dict[str, np.ndarray] = {}

    for k, y in Y_train.items():
        if not scale_y:
            Y_train_scaled[k] = y.astype("float32")
            continue
        ys = StandardScaler()
        yf = y.reshape(-1, 1)
        y_scaled = ys.fit_transform(yf).reshape(y.shape).astype("float32")
        y_scalers[k] = ys
        Y_train_scaled[k] = y_scaled

    return X_train_scaled, Y_train_scaled, Scalers(x_scaler=x_scaler, y_scalers=y_scalers)


def transform_with_scalers(
    X: np.ndarray,
    Y: Dict[str, np.ndarray],
    scalers: Scalers,
    scale_y: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    Xf = X.reshape(-1, X.shape[-1])
    X_scaled = scalers.x_scaler.transform(Xf).reshape(X.shape).astype("float32")

    Y_scaled = {}
    for k, y in Y.items():
        if not scale_y:
            Y_scaled[k] = y.astype("float32")
            continue
        ys = scalers.y_scalers[k]
        y_scaled = ys.transform(y.reshape(-1, 1)).reshape(y.shape).astype("float32")
        Y_scaled[k] = y_scaled
    return X_scaled, Y_scaled


def inverse_transform_y(
    Y_scaled: Dict[str, np.ndarray],
    scalers: Scalers,
    scale_y: bool = True,
) -> Dict[str, np.ndarray]:
    if not scale_y:
        return {k: v.copy() for k, v in Y_scaled.items()}
    out = {}
    for k, y in Y_scaled.items():
        ys = scalers.y_scalers[k]
        out[k] = ys.inverse_transform(y.reshape(-1, 1)).reshape(y.shape).astype("float32")
    return out


# -------------------------
# Model building
# -------------------------
def build_multitask_model(
    n_input_steps: int,
    n_features: int,
    n_output_steps: int,
    targets: List[str],
    rnn_type: str = "LSTM",
    rnn_units: int = 64,
    n_layers: int = 2,
    dense_units: int = 64,
    dropout: float = 0.2,
) -> tf.keras.Model:
    rnn_type = rnn_type.upper()
    RNN = {"LSTM": layers.LSTM, "GRU": layers.GRU, "SIMPLERNN": layers.SimpleRNN}.get(rnn_type)
    if RNN is None:
        raise ValueError(f"Unsupported rnn_type={rnn_type}. Choose from LSTM, GRU, SimpleRNN.")

    inp = layers.Input(shape=(n_input_steps, n_features), name="X")

    x = inp
    for i in range(n_layers):
        x = RNN(rnn_units, return_sequences=True, name=f"{rnn_type.lower()}_{i+1}")(x)
        x = layers.LayerNormalization(name=f"ln_{i+1}")(x)

    gap = layers.GlobalAveragePooling1D(name="gap")(x)
    gmp = layers.GlobalMaxPooling1D(name="gmp")(x)
    x = layers.Concatenate(name="pool_concat")([gap, gmp])

    x = layers.Dense(dense_units, activation="relu", name="shared_dense")(x)
    x = layers.Dropout(dropout, name="shared_dropout")(x)

    outputs = {}
    for t in targets:
        # All targets as sequences over horizon (consistent evaluation)
        outputs[t] = layers.Dense(n_output_steps, name=t)(x)

    model = models.Model(inputs=inp, outputs=outputs, name="cloud_multitask")
    return model


def make_losses(loss_name: str, targets: List[str]) -> Dict[str, tf.keras.losses.Loss]:
    lname = loss_name.lower()
    if lname in ("mse", "mean_squared_error"):
        base = tf.keras.losses.MeanSquaredError()
    elif lname in ("mae", "mean_absolute_error"):
        base = tf.keras.losses.MeanAbsoluteError()
    elif lname in ("huber",):
        base = tf.keras.losses.Huber()
    else:
        raise ValueError(f"Unsupported loss '{loss_name}'. Choose from mse, mae, huber.")
    return {t: base for t in targets}


def parse_loss_weights(s: str, targets: List[str]) -> Dict[str, float]:
    """
    Example: "rain_gsp_rate_L00=1,cloud_base=0.5,tqc_L00=0.2"
    If empty -> all 1.0
    """
    if not s:
        return {t: 1.0 for t in targets}
    out = {t: 1.0 for t in targets}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = float(v.strip())
        if k not in out:
            raise KeyError(f"Loss weight key '{k}' not in targets: {targets}")
        out[k] = v
    return out


# -------------------------
# Baseline & metrics
# -------------------------
def persistence_baseline(
    X_scaled: np.ndarray,
    Y_scaled: Dict[str, np.ndarray],
    meta: dict,
    targets: List[str],
    n_output_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Persistence baseline: predict future values as last input value (repeated).
    Works for targets that are included as input scalar features or ts features.
    For safety, if a target is not in X_feature_names, fallback to predicting zeros.
    """
    feat_names = meta["X_feature_names"]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    last_step = X_scaled[:, -1, :]  # (N, F)

    preds = {}
    for t in targets:
        if t in name_to_idx:
            v = last_step[:, name_to_idx[t]]  # (N,)
            preds[t] = np.repeat(v[:, None], n_output_steps, axis=1).astype("float32")
        else:
            preds[t] = np.zeros((X_scaled.shape[0], n_output_steps), dtype="float32")
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # shapes (N, H)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    # per horizon
    mae_h = np.mean(np.abs(err), axis=0).astype("float64")
    rmse_h = np.sqrt(np.mean(err ** 2, axis=0)).astype("float64")
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mae_by_horizon": mae_h.tolist(),
        "rmse_by_horizon": rmse_h.tolist(),
    }


def compute_skill(model_metrics: dict, baseline_metrics: dict) -> dict:
    # skill = 1 - (rmse_model / rmse_baseline)
    out = {}
    for t in model_metrics.keys():
        rmse_m = model_metrics[t]["rmse"]
        rmse_b = baseline_metrics[t]["rmse"]
        out[t] = {
            "skill_rmse": float(1.0 - (rmse_m / rmse_b)) if rmse_b > 0 else None
        }
    return out


# -------------------------
# Plotting
# -------------------------
def plot_history(history: tf.keras.callbacks.History, outdir: Path) -> None:
    ensure_dir(outdir)
    hist = history.history if history is not None else {}
    if not hist:
        return

    # loss
    plt.figure()
    if "loss" in hist:
        plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()

    # per-task metrics if present
    # (Keras logs e.g. rain_gsp_rate_L00_loss, val_rain_gsp_rate_L00_loss)
    for k in list(hist.keys()):
        if k.endswith("_loss") and not k.startswith("val_") and k != "loss":
            task = k[:-5]
            plt.figure()
            plt.plot(hist[k], label=f"train_{k}")
            val_key = "val_" + k
            if val_key in hist:
                plt.plot(hist[val_key], label=f"val_{k}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"loss_curve_{task}.png", dpi=150)
            plt.close()


def plot_horizon_curves(metrics: dict, outdir: Path, prefix: str) -> None:
    ensure_dir(outdir)
    for task, m in metrics.items():
        for key in ["mae_by_horizon", "rmse_by_horizon"]:
            arr = np.array(m[key], dtype=float)
            plt.figure()
            plt.plot(arr)
            plt.xlabel("Horizon step")
            plt.ylabel(key.replace("_by_horizon", "").upper())
            plt.tight_layout()
            plt.savefig(outdir / f"{prefix}_{key}_{task}.png", dpi=150)
            plt.close()


def plot_forecast_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: Path,
    task: str,
    n_examples: int = 5,
) -> None:
    ensure_dir(outdir)
    n = min(n_examples, y_true.shape[0])
    idxs = np.linspace(0, y_true.shape[0] - 1, n, dtype=int)
    for j, i in enumerate(idxs):
        plt.figure()
        plt.plot(y_true[i], label="true")
        plt.plot(y_pred[i], label="pred")
        plt.xlabel("Horizon step")
        plt.ylabel(task)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"forecast_example_{task}_{j}.png", dpi=150)
        plt.close()


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: Path,
    task: str,
) -> None:
    ensure_dir(outdir)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    plt.figure()
    plt.scatter(yt, yp, s=4)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{task}.png", dpi=150)
    plt.close()

    # residual histogram
    resid = yp - yt
    plt.figure()
    plt.hist(resid, bins=60)
    plt.xlabel("residual (pred - true)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / f"residual_hist_{task}.png", dpi=150)
    plt.close()


# -------------------------
# HF upload (optional)
# -------------------------
def hf_maybe_login(token: Optional[str]) -> None:
    if not HF_AVAILABLE:
        raise RuntimeError("huggingface_hub not available")
    if token:
        hf_login(token=token)
    else:
        # huggingface_hub will also look for HF_TOKEN in env for some operations;
        # we don't force login if not provided.
        pass


def hf_upload_folder(local_dir: Path, repo_id: str, remote_dir: str, token: Optional[str]) -> None:
    if not HF_AVAILABLE:
        raise RuntimeError("huggingface_hub not available")
    hf_maybe_login(token)
    api = HfApi(token=token)

    # upload_folder exists in newer versions; fall back to per-file upload if needed
    if hasattr(api, "upload_folder"):
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=remote_dir.rstrip("/"),
        )
        return

    # fallback: per file upload
    for p in local_dir.rglob("*"):
        if p.is_file():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=str(Path(remote_dir) / p.relative_to(local_dir)).replace("\\", "/"),
                repo_id=repo_id,
                repo_type="dataset",
            )


# -------------------------
# Main pipeline
# -------------------------
def load_samples(
    data_source: str,
    repo_id: Optional[str],
    repo_subdir: str,
    local_data_path: Optional[Path],
    data_glob: str,
    track_len_csv_in_repo: Optional[str],
    track_len_csv_local: Optional[Path],
    min_track_len: Optional[int],
    datashare: float,
    seed: int,
    cutoff_steps: int,
    scalar_features: List[str],
    step_seconds: int,
    lon_col: str,
    lat_col: str,
    add_motion: bool,
    add_scalar_ratios: bool,
    add_hydrometeor_stats: bool,
) -> Tuple[List[Dict[str, np.ndarray]], dict]:
    """
    data_source: 'hf' or 'local' or 'hf+local'
    """
    files_hf: List[str] = []
    files_local: List[Path] = []
    if data_source in ("hf", "hf+local"):
        if not repo_id:
            raise ValueError("--repo-id is required for --data-source hf")
        files_hf = list_hf_csv_files(
            repo_id=repo_id,
            repo_subdir=repo_subdir,
            track_len_csv_in_repo=track_len_csv_in_repo,
            min_track_len=min_track_len,
            datashare=datashare,
            seed=seed,
        )

    if data_source in ("local", "hf+local"):
        if local_data_path is None:
            raise ValueError("--local-data-path is required for --data-source local")
        files_local = list_local_csv_files(
            local_data_path=local_data_path,
            data_glob=data_glob,
            track_len_csv_local=track_len_csv_local,
            min_track_len=min_track_len,
            datashare=datashare,
            seed=seed,
        )

    samples: List[Dict[str, np.ndarray]] = []
    used = {"hf": [], "local": []}

    # HF files
    for f in files_hf:
        try:
            local_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f)
            df = pd.read_csv(local_file)
            if len(df) <= cutoff_steps:
                continue
            sample = preprocess_cloud(
                df,
                scalar_features=scalar_features.copy(),
                step_seconds=step_seconds,
                lon_col=lon_col,
                lat_col=lat_col,
                add_motion=add_motion,
                add_scalar_ratios=add_scalar_ratios,
                add_hydrometeor_stats=add_hydrometeor_stats,
            )
            samples.append(sample)
            used["hf"].append(f)
        except Exception as e:
            print(f"[WARN] Skipping HF file {f}: {e}", file=sys.stderr)

    # Local files
    for p in files_local:
        try:
            df = pd.read_csv(p)
            if len(df) <= cutoff_steps:
                continue
            sample = preprocess_cloud(
                df,
                scalar_features=scalar_features.copy(),
                step_seconds=step_seconds,
                lon_col=lon_col,
                lat_col=lat_col,
                add_motion=add_motion,
                add_scalar_ratios=add_scalar_ratios,
                add_hydrometeor_stats=add_hydrometeor_stats,
            )
            samples.append(sample)
            used["local"].append(str(p))
        except Exception as e:
            print(f"[WARN] Skipping local file {p}: {e}", file=sys.stderr)

    info = {
        "n_samples": len(samples),
        "used_files": used,
    }
    return samples, info


def main():
    ap = argparse.ArgumentParser(description="Train & evaluate multi-task cloud evolution model.")

    # data
    ap.add_argument("--data-source", choices=["hf", "local", "hf+local"], default="hf",
                    help="Where to load data from.")
    ap.add_argument("--repo-id", type=str, default="mttfst/Paulette_Cloud_Tracks",
                    help="HF dataset repo id (only for hf/hf+local).")
    ap.add_argument("--repo-subdir", type=str, default="exp_1.1",
                    help="Subdir in HF dataset repo containing CSVs.")
    ap.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""),
                    help="HF token. Alternatively set HF_TOKEN env var.")
    ap.add_argument("--local-data-path", type=str, default="",
                    help="Local root directory containing CSV files (only for local/hf+local).")
    ap.add_argument("--data-glob", type=str, default="exp_1.1/*.csv",
                    help="Glob relative to local-data-path used to find CSV files.")
    ap.add_argument("--track-len-csv-in-repo", type=str, default="track_len/track_len_exp_1.1.csv",
                    help="Track length CSV path inside HF repo (set empty to disable).")
    ap.add_argument("--track-len-csv-local", type=str, default="",
                    help="Local track length CSV (set empty to disable).")
    ap.add_argument("--min-track-len", type=int, default=120,
                    help="Minimum track length to include (set 0 to disable).")
    ap.add_argument("--datashare", type=float, default=1.0, help="Fraction of available tracks to use.")
    ap.add_argument("--cutoff-steps", type=int, default=5, help="Skip tracks with <= cutoff steps.")

    # time/window
    ap.add_argument("--step-seconds", type=int, default=30, help="Time resolution per step in seconds.")
    ap.add_argument("--input-minutes", type=int, default=30, help="Input history length in minutes.")
    ap.add_argument("--output-minutes", type=int, default=10, help="Forecast horizon in minutes.")
    ap.add_argument("--stride-minutes", type=int, default=None,
                    help="Stride in minutes (defaults to output-minutes).")
    ap.add_argument("--stride-steps", type=int, default=None,
                    help="Stride in steps (overrides stride-minutes if provided).")

    # motion features
    ap.add_argument("--add-motion-features", action="store_true", help="Add motion features from lon/lat.")
    ap.add_argument("--lon-col", type=str, default="lon", help="Longitude column name in raw data.")
    ap.add_argument("--lat-col", type=str, default="lat", help="Latitude column name in raw data.")

    # engineered features
    ap.add_argument("--add-scalar-ratios", action="store_true", help="Add scalar ratios tqc/tqi and tqc/rainrate.")
    ap.add_argument("--add-hydrometeor-stats", action="store_true",
                    help="Add profile stats (max/height_max/sum) for qc/qi/qr/qg/qs.")

    # targets
    ap.add_argument("--targets", type=str, default="rain_gsp_rate_L00,cloud_base,tqc_L00,tqi_L00",
                    help="Comma-separated targets. Each must be in scalars or ts_features. "
                         "Use 'rul' as alias for remaining_lifetime column.")

    # training
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    ap.add_argument("--loss", type=str, default="mse", help="Loss: mse, mae, huber.")
    ap.add_argument("--loss-weights", type=str, default="",
                    help="Optional per-target loss weights, e.g. 'rain_gsp_rate_L00=1,cloud_base=0.5'")

    # model
    ap.add_argument("--rnn-type", type=str, default="LSTM", choices=["LSTM", "GRU", "SimpleRNN"])
    ap.add_argument("--rnn-units", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dense-units", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)

    # scaling
    ap.add_argument("--scale-y", action="store_true", help="Scale targets with StandardScaler (recommended).")

    # output
    ap.add_argument("--output-root", type=str, default="runs_local", help="Local output root folder.")
    ap.add_argument("--run-name", type=str, default="", help="Optional run name. If empty, auto-generated.")
    ap.add_argument("--save-n-examples", type=int, default=5, help="How many forecast example plots to save per task.")

    # HF upload outputs (optional)
    ap.add_argument("--push-to-hf", action="store_true", help="Upload local output folder to HF dataset repo.")
    ap.add_argument("--hf-output-dir", type=str, default="runs/MF", help="Remote output dir in HF dataset repo.")

    args = ap.parse_args()

    set_global_seed(args.seed)

    # windows
    n_input_steps = int(args.input_minutes * 60 / args.step_seconds)
    n_output_steps = int(args.output_minutes * 60 / args.step_seconds)

    stride_minutes = args.stride_minutes if args.stride_minutes is not None else args.output_minutes
    stride_steps = args.stride_steps if args.stride_steps is not None else int(stride_minutes * 60 / args.step_seconds)
    stride_steps = max(1, int(stride_steps))

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        raise ValueError("No targets specified")

    # scalar features: base from notebook (already removed lwp/iwp)
    scalar_features = list(SCALAR_FEATURES_BASE)

    local_data_path = Path(args.local_data_path) if args.local_data_path else None
    track_len_csv_local = Path(args.track_len_csv_local) if args.track_len_csv_local else None
    min_track_len = args.min_track_len if args.min_track_len and args.min_track_len > 0 else None
    track_len_csv_in_repo = args.track_len_csv_in_repo if args.track_len_csv_in_repo else None

    # output dirs
    run_name = args.run_name.strip()
    if not run_name:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"cloud_multitask_{ts}"
    out_root = Path(args.output_root)
    run_dir = ensure_dir(out_root / run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    preds_dir = ensure_dir(run_dir / "predictions")
    scalers_dir = ensure_dir(run_dir / "scalers")

    # HF login only if needed
    if args.data_source in ("hf", "hf+local") or args.push_to_hf:
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed; cannot use hf data-source or push-to-hf.")
        if args.hf_token:
            hf_maybe_login(args.hf_token)

    # load samples
    samples, load_info = load_samples(
        data_source=args.data_source,
        repo_id=args.repo_id,
        repo_subdir=args.repo_subdir,
        local_data_path=local_data_path,
        data_glob=args.data_glob,
        track_len_csv_in_repo=track_len_csv_in_repo,
        track_len_csv_local=track_len_csv_local,
        min_track_len=min_track_len,
        datashare=args.datashare,
        seed=args.seed,
        cutoff_steps=args.cutoff_steps,
        scalar_features=scalar_features,
        step_seconds=args.step_seconds,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        add_motion=args.add_motion_features,
        add_scalar_ratios=args.add_scalar_ratios,
        add_hydrometeor_stats=args.add_hydrometeor_stats,
    )
    print(f"Loaded samples: {load_info['n_samples']}")

    # split by track/file (already split by file listing)
    # Since we loaded into a single list (from multiple sources), we split on sample index deterministically.
    idxs = list(range(len(samples)))
    train_idx, val_idx, test_idx = split_files(idxs, seed=args.seed, train_frac=0.7, val_frac=0.15)

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    test_samples  = [samples[i] for i in test_idx]

    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # windows
    X_train, Y_train, meta = build_windows_multitask(
        train_samples, n_input_steps=n_input_steps, n_output_steps=n_output_steps,
        stride=stride_steps, targets=targets
    )
    X_val, Y_val, _ = build_windows_multitask(
        val_samples, n_input_steps=n_input_steps, n_output_steps=n_output_steps,
        stride=stride_steps, targets=targets
    )
    X_test, Y_test, _ = build_windows_multitask(
        test_samples, n_input_steps=n_input_steps, n_output_steps=n_output_steps,
        stride=stride_steps, targets=targets
    )

    # scaling
    X_train_scaled, Y_train_scaled, scalers = fit_transform_scalers(X_train, Y_train, scale_y=args.scale_y)
    X_val_scaled, Y_val_scaled = transform_with_scalers(X_val, Y_val, scalers, scale_y=args.scale_y)
    X_test_scaled, Y_test_scaled = transform_with_scalers(X_test, Y_test, scalers, scale_y=args.scale_y)

    # save scalers + feature meta
    joblib.dump(scalers.x_scaler, scalers_dir / "x_scaler.pkl")
    for k, ys in scalers.y_scalers.items():
        joblib.dump(ys, scalers_dir / f"y_scaler_{k}.pkl")

    save_json(run_dir / "feature_meta.json", meta)

    # model
    model = build_multitask_model(
        n_input_steps=n_input_steps,
        n_features=X_train_scaled.shape[-1],
        n_output_steps=n_output_steps,
        targets=targets,
        rnn_type=args.rnn_type,
        rnn_units=args.rnn_units,
        n_layers=args.n_layers,
        dense_units=args.dense_units,
        dropout=args.dropout,
    )

    # compile
    losses = make_losses(args.loss, targets)
    loss_weights = parse_loss_weights(args.loss_weights, targets)

    opt_name = args.optimizer.lower()
    if opt_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    elif opt_name == "adamw":
        opt = tf.keras.optimizers.AdamW(learning_rate=args.learning_rate)
    elif opt_name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)

    # save model summary
    summary_lines = []
    model.summary(print_fn=lambda s: summary_lines.append(s))
    save_text(run_dir / "model_summary.txt", "\n".join(summary_lines))

    # callbacks
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "model_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # train
    history = model.fit(
        X_train_scaled,
        Y_train_scaled,
        validation_data=(X_val_scaled, Y_val_scaled),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=2,
    )

    # save final model
    model.save(run_dir / "model_final.keras")

    # save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(run_dir / "history.csv", index=False)

    # plots: history
    plot_history(history, plots_dir)

    # predict
    Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    # Keras returns dict of arrays
    Y_pred_scaled = {k: np.array(v, dtype="float32") for k, v in Y_pred_scaled.items()}

    # baseline (scaled space)
    Y_base_scaled = persistence_baseline(X_test_scaled, Y_test_scaled, meta, targets, n_output_steps)

    # inverse to physical space if scaled
    Y_true = inverse_transform_y(Y_test_scaled, scalers, scale_y=args.scale_y)
    Y_pred = inverse_transform_y(Y_pred_scaled, scalers, scale_y=args.scale_y)
    Y_base = inverse_transform_y(Y_base_scaled, scalers, scale_y=args.scale_y)

    # save predictions
    for t in targets:
        np.save(preds_dir / f"y_true_{t}.npy", Y_true[t])
        np.save(preds_dir / f"y_pred_{t}.npy", Y_pred[t])
        np.save(preds_dir / f"y_base_{t}.npy", Y_base[t])

    # metrics
    model_metrics = {t: compute_metrics(Y_true[t], Y_pred[t]) for t in targets}
    baseline_metrics = {t: compute_metrics(Y_true[t], Y_base[t]) for t in targets}
    skill = compute_skill(model_metrics, baseline_metrics)

    metrics_bundle = {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "skill": skill,
    }
    save_json(run_dir / "metrics.json", metrics_bundle)

    # plots: horizon curves
    plot_horizon_curves(model_metrics, plots_dir, prefix="model")
    plot_horizon_curves(baseline_metrics, plots_dir, prefix="baseline")

    # plots: examples & scatter
    for t in targets:
        plot_forecast_examples(Y_true[t], Y_pred[t], plots_dir, task=t, n_examples=args.save_n_examples)
        plot_scatter(Y_true[t], Y_pred[t], plots_dir, task=t)

    # config snapshot (everything needed to reproduce)
    config = {
        "run_name": run_name,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "data": {
            "data_source": args.data_source,
            "repo_id": args.repo_id,
            "repo_subdir": args.repo_subdir,
            "local_data_path": str(local_data_path) if local_data_path else None,
            "data_glob": args.data_glob,
            "track_len_csv_in_repo": track_len_csv_in_repo,
            "track_len_csv_local": str(track_len_csv_local) if track_len_csv_local else None,
            "min_track_len": min_track_len,
            "datashare": args.datashare,
            "cutoff_steps": args.cutoff_steps,
            "load_info": load_info,
        },
        "time": {
            "step_seconds": args.step_seconds,
            "input_minutes": args.input_minutes,
            "output_minutes": args.output_minutes,
            "n_input_steps": n_input_steps,
            "n_output_steps": n_output_steps,
            "stride_steps": stride_steps,
        },
        "features": {
            "scalar_features_base": SCALAR_FEATURES_BASE,
            "add_scalar_ratios": args.add_scalar_ratios,
            "add_motion_features": args.add_motion_features,
            "lon_col": args.lon_col,
            "lat_col": args.lat_col,
            "add_hydrometeor_stats": args.add_hydrometeor_stats,
            "X_feature_names": meta["X_feature_names"],
            "scalar_input_features": meta["scalar_input_features"],
            "ts_feature_names": meta["ts_feature_names"],
            "rul_feature_name_in_scalars": meta["rul_feature_name_in_scalars"],
        },
        "targets": targets,
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "loss": args.loss,
            "loss_weights": loss_weights,
            "scale_y": bool(args.scale_y),
        },
        "model": {
            "rnn_type": args.rnn_type,
            "rnn_units": args.rnn_units,
            "n_layers": args.n_layers,
            "dense_units": args.dense_units,
            "dropout": args.dropout,
        },
        "tensorflow": tf.__version__,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    save_json(run_dir / "config.json", config)

    print(f"✅ Finished. Outputs saved to: {run_dir}")

    # optional HF push
    if args.push_to_hf:
        remote_dir = f"{args.hf_output_dir.rstrip('/')}/{run_name}"
        print(f"⬆️  Uploading outputs to HF: {args.repo_id} -> {remote_dir}")
        hf_upload_folder(run_dir, repo_id=args.repo_id, remote_dir=remote_dir, token=args.hf_token or None)
        print("✅ Upload finished.")

if __name__ == "__main__":
    main()
