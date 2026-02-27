
# Cloud Multi‑Task Forecasting Framework

---

# 1. Project Overview

This framework implements a configurable multi‑task deep learning system for forecasting the future evolution of tracked cloud systems.

Targets predicted simultaneously:

- Surface rain rate (`rain_gsp_rate_L00`)
- Cloud base height (`cloud_base`)
- Total cloud liquid water (`tqc_L00`)
- Total cloud ice (`tqi_L00`)

All targets are forecast as **future sequences**, enabling horizon‑aware evaluation and physically interpretable analysis.

The framework is designed for reproducible HPC experimentation using SLURM job arrays.

---

# 2. Evaluation Philosophy

## Persistence Baseline

All results are evaluated relative to a **persistence model**:

```
ŷ(t + Δt) = y(t)
```

The baseline assumes the cloud state does not evolve.

### Why persistence?

- Strong physical short‑term forecast baseline
- Independent of ML assumptions
- Defines minimum useful predictive skill
- Allows comparison across variables with different units

A model must outperform persistence to demonstrate learned dynamics.

---

## Skill Definition

Performance is reported using RMSE‑based skill:

```
Skill = 1 − RMSE_model / RMSE_persistence
```

Interpretation:

| Skill | Meaning |
|---|---|
| > 0 | Better than persistence |
| = 0 | Equal to persistence |
| < 0 | Worse than persistence |

This metric is dimensionless and comparable across targets.

---

# 3. Input Feature System

## Scalar Features
- cape_ml_L00
- cin_ml_L00
- rain_gsp_rate_L00
- tqc_L00
- tqi_L00
- area_m2

Engineered ratios:
- tqc / tqi
- tqc / rain rate

## Profile‑Derived Diagnostics
Profiles: qc, qi, qr, qg, qs, qv, roh, w

Derived quantities include:
- cloud base/top/thickness
- vertical velocity statistics
- mass integrals
- height of maxima
- center of mass

## Motion Features
Derived from relative lon/lat displacement:
dx, dy, speed, heading (sin/cos), acceleration, turn rate.

Absolute position is excluded to avoid geographic leakage.

---

# 4. Model Architecture

Shared encoder:

```
Input → GRU/LSTM → LayerNorm
      → Global Avg + Max Pool
      → Dense → Dropout
```

Separate dense heads predict each target sequence.

Final architecture selection favors shallow recurrent models.

---

# 5. Experimental Progression

---

## Phase 1 — Structural Model Selection

### Objective
Identify meaningful feature groups and architectures before tuning.

### Fixed Parameters
- datashare = 0.5
- epochs = 20
- input window = 30 min
- horizon = 10 min
- stride = 10 min
- units = 64
- layers = 2

### Swept Dimensions

| Category | Values |
|---|---|
| Feature Sets | F0 (ratios), F1 (ratios+motion), F2 (ratios+motion+hydro) |
| Architecture | GRU, LSTM |
| Loss | MSE, Huber |

Total runs: **12**

### Mean Skill Results

| Target | Mean Skill |
|---|---|
| Rain | 0.118 |
| Cloud base | 0.053 |
| TQC | -0.092 |
| TQI | -0.123 |

### Key Findings
- Motion features improve rain prediction.
- GRU outperforms LSTM.
- Hydrometeor statistics add no benefit.
- Microphysics largely persistence dominated.

Baseline established:
**GRU + Motion (F1) + MSE**.

---

## Phase 2 — Parameter Sensitivity

### Objective
Understand which parameter groups control performance.

### Baseline Configuration
GRU, 64 units, 2 layers, input 30 min, horizon 10 min.

### Sweeps

#### Sweep A — Capacity
Units: 32, 64, 128, 256  
Layers: 1–4  
Runs: 16

Result:
- Best ≈ 1 layer, wider networks.
- Skill ≈ 0.10 rain.

#### Sweep B — Temporal Representation
Input: 20, 30, 60, 90 min  
Horizon: 5, 10, 20 min  
Stride: overlap / non‑overlap  
Runs: 24

Best skill:
- Rain ≈ **0.17**
- Input 20–30 min
- Horizon 20 min

#### Sweep C — Optimization
Learning rate: 5e‑4, 1e‑3, 2e‑3  
Dropout: 0.1–0.3  
Batch: 32–128  
Runs: 27

Effect size small (≈0.01 skill).

### Phase 2 Conclusion

Temporal alignment dominates predictive skill.

Predictability timescale ≈ **20 minutes**.

---

## Phase 3 — Integration

### Objective
Combine Phase‑2 winners and test robustness.

### Tested Grid
- Units: 128, 256
- Input: 20, 30 min
- Horizon: 20 min
- Seeds: 2

Total runs: 8

### Mean Skill Results

| Configuration | Rain Skill | Cloud Skill |
|---|---|---|
| 128 units + 20 min | **0.169** | 0.135 |
| 256 units + 30 min | 0.165 | 0.136 |
| 256 units + 20 min | 0.163 | 0.125 |
| 128 units + 30 min | 0.161 | 0.129 |

### Findings
- Capacity saturation reached.
- Temporal setup remains dominant.
- Stable across seeds.

Selected configuration:
GRU, 128 units, 1 layer, input 20 min, horizon 20 min.

---

## Phase 4 — Final Training (Full Dataset)

### Configuration
- Datashare = 1.0
- Epochs = 40
- Seed = 42

### Final Skill (vs Persistence)

| Target | Skill |
|---|---|
| Rain | **0.218** |
| Cloud base | **0.161** |
| TQC | **0.144** |
| TQI | **0.082** |
| Mean | **0.151** |

### Interpretation
- Skill increases with dataset size.
- Rain prediction shows strongest dynamical signal.
- Microphysics partially predictable once temporal scale is correct.

---

# 6. Scientific Conclusions

Across all phases:

1. Forecast skill is governed primarily by **temporal alignment**.
2. Moderate model capacity is sufficient.
3. Motion features encode essential cloud dynamics.
4. Microphysical quantities remain closer to persistence behaviour.
5. Effective cloud predictability timescale ≈ **20 minutes**.

---

# 7. Final Model

Final Model configuration:

| Component | Setting |
|---|---|
| Architecture | GRU |
| Layers | 1 |
| Units | 128 |
| Input | 20 min |
| Horizon | 20 min |
| Stride | 10 min |
| Loss | MSE |
| Learning Rate | 0.002 |
| Dropout | 0.1 |
| Batch Size | 32 |
| Features | Ratios + Motion |

