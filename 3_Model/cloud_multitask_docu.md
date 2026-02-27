
# Cloud Multi‑Task Forecasting Framework
## Consolidated Technical Documentation

---

# 1. Project Overview

This framework implements a configurable multi‑task deep learning system for forecasting the future evolution of tracked cloud systems.  
The goal is **physically interpretable cloud evolution prediction**, not single‑variable regression.

The model jointly predicts future sequences of:

- Surface rain rate
- Cloud base height
- Total cloud liquid water (TQC)
- Total cloud ice (TQI)

All experiments are evaluated relative to a **persistence baseline**, enabling physically meaningful skill assessment.

---

# 2. Model Philosophy

The framework treats cloud evolution as a dynamical forecasting problem.

Key principles:

- Multi‑task prediction enforces physical consistency.
- Temporal alignment is more important than model complexity.
- Evaluation must compare against persistence rather than absolute error.
- Experiments are designed for reproducible HPC sweeps.

---

# 3. Input Features

## 3.1 Scalar Inputs
- cape_ml_L00
- cin_ml_L00
- rain_gsp_rate_L00
- tqc_L00
- tqi_L00
- area_m2

Engineered ratios:
- tqc / tqi
- tqc / rainrate (rain efficiency)

`remaining_lifetime` is used only as a label and excluded from inputs.

## 3.2 Profile‑Derived Features

Vertical profiles used:
qc, qi, qr, qg, qs, qv, roh, w

Derived diagnostics include:
- cloud base/top/thickness
- mass integrals
- vertical velocity statistics
- height of maxima
- center of mass

## 3.3 Motion Features

Derived from relative lon/lat displacement:

- dx, dy
- speed
- heading (sin/cos)
- acceleration
- turn rate

Absolute position is intentionally excluded.

---

# 4. Model Architecture

Shared encoder:

```
Input → GRU/LSTM → LayerNorm
      → Global Avg + Max Pool
      → Dense → Dropout
```

Separate output heads predict full future sequences for each target.

Final selected architecture:
- GRU
- shallow (1 layer)
- moderate width

---

# 5. Training & Evaluation

## Training
- Early stopping
- LR scheduling
- Checkpointing
- Feature scaling

## Evaluation Metrics
- MAE
- RMSE
- Bias
- Horizon‑wise errors
- Persistence skill score

Skill definition:

```
Skill = 1 − RMSE_model / RMSE_persistence
```

All runs store configs, metrics, predictions, and plots for reproducibility.

---

# 6. HPC Design

The training script is fully CLI‑configurable and designed for SLURM arrays:

- isolated run directories
- reproducible config snapshots
- large parameter sweeps
- no shared state conflicts

---

# 7. Experimental Progression

## Phase 1 — Structural Model Selection

Goal: identify meaningful feature groups and architectures.

Tested:
- Feature complexity (F0/F1/F2)
- GRU vs LSTM
- MSE vs Huber loss

**Result**
- Motion features essential.
- GRU outperforms LSTM.
- Hydrometeor statistics add no benefit.
- Rain shows strongest predictability.

Established baseline:
GRU + Motion (F1) + MSE.

---

## Phase 2 — Parameter Sensitivity

Goal: understand *why* the model works.

Three isolated sweeps:

| Sweep | Purpose |
|---|---|
| Capacity | model size scaling |
| Temporal | window & horizon |
| Optimization | training stability |

**Key Finding**

Temporal configuration dominates performance.

Best regime:
- input ≈ 20–30 min
- forecast horizon ≈ 20 min

Predictability is governed by cloud dynamics rather than network size.

---

## Phase 3 — Integration

Combined Phase‑2 winners:

- Units: 128–256
- Layers: 1
- Input: 20–30 min
- Horizon: 20 min
- Optim: lr=0.002, dropout=0.1, batch=32

Result:
- Stable performance across seeds
- Capacity saturation observed
- Temporal alignment confirmed as dominant factor

Selected final configuration:

| Component | Choice |
|---|---|
| Architecture | GRU |
| Layers | 1 |
| Units | 128 |
| Input | 20 min |
| Horizon | 20 min |
| Stride | 10 min |
| Features | Ratios + Motion |
| Loss | MSE |

---

# 8. Final Model (Full Dataset)

The selected configuration was retrained using the complete dataset.

## Configuration
- Datashare: 1.0
- Epochs: 40
- Seed: 42

## Performance (vs Persistence)

| Target | Skill |
|---|---|
| Rain | **0.218** |
| Cloud base | **0.161** |
| TQC | **0.144** |
| TQI | **0.082** |
| Mean | **0.151** |

---

# 9. Scientific Conclusions

Across all phases:

1. Forecast skill is controlled primarily by **temporal alignment**.
2. Moderate model capacity is sufficient.
3. Motion information captures essential cloud dynamics.
4. Microphysics remains partially persistence‑dominated.
5. Increasing data volume improves stability and skill.

The experiments empirically identify an effective cloud predictability timescale of ~20 minutes.

---



