# Model Definition and Evaluation

**[Code](cloud_multitask_train.py)**



# 1. Overview

This document describes the definition, implementation, and evaluation of the machine learning model used to predict short‑term cloud evolution in simulated hurricane environments.

The framework forecasts future sequences of:

- Surface rain rate
- Cloud base height
- Total cloud liquid water (TQC)
- Total cloud ice (TQI)

The objective is to determine whether cloud dynamics can be predicted from the recent physical state and motion history of tracked cloud systems.



# 2. Evaluation Philosophy

## Persistence Baseline

All results are evaluated relative to a persistence forecast:

ŷ(t + Δt) = y(t)

The baseline assumes that the cloud state does not change.

### Motivation

Persistence represents a strong physical short‑term forecast. A model must outperform persistence to demonstrate learned dynamics rather than simple state reproduction.

## Skill Definition

Skill is defined as:

Skill = 1 − RMSE_model / RMSE_persistence

Interpretation:

| Skill | Meaning |
|---|---|
| > 0 | Better than persistence |
| = 0 | Equal to persistence |
| < 0 | Worse than persistence |



# 3. Feature Engineering

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

## Profile‑Derived Features

Vertical profiles:

qc, qi, qr, qg, qs, qv, roh, w

Derived diagnostics:

- cloud base/top/thickness
- mass integrals
- vertical velocity statistics
- height of maxima
- center of mass

## Motion Features

Derived from relative lon/lat displacement:

dx, dy, speed, heading, acceleration, turn rate.

Absolute geographic position is excluded.



# 4. Model Architecture

Shared encoder:

Input → RNN (GRU/LSTM) → LayerNorm  
→ Global Avg + Max Pool → Dense → Dropout

Separate output heads predict future sequences for each target.



# 4.1 Model Selection

Cloud evolution is inherently sequential, motivating recurrent neural networks.

Two architectures were evaluated:

- LSTM (Long Short‑Term Memory)
- GRU (Gated Recurrent Unit)

### Motivation

| Property | LSTM | GRU |
|---|---|---|
| Parameter count | higher | lower |
| Computational cost | higher | lower |
| Memory mechanism | explicit gates | simplified gating |

Because cloud dynamics occur on short temporal scales, both were tested experimentally.

### Result

Phase‑1 experiments showed:

- GRU achieved higher skill scores,
- training was more stable,
- computational efficiency improved.

Therefore GRU was selected as the final architecture.



# 5. Training and Evaluation

Training includes:

- Early stopping
- Learning rate scheduling
- Model checkpointing
- Feature scaling

Metrics:

- MAE
- RMSE
- Bias
- Horizon‑wise errors
- Persistence skill score

All runs store predictions, metrics, configuration snapshots, and plots.



# 6. HPC‑Focused Implementation

## 6.1 Implementation

The final system is implemented as a standalone Python training framework:

**[cloud_multitask_train.py](cloud_multitask_train.py)**

The transition from notebooks to a Python script was required for:

- large hyperparameter sweeps,
- reproducibility,
- execution on HPC infrastructure,
- SLURM job arrays.

Key properties:

- fully CLI‑configurable experiments
- reproducible configuration snapshots
- isolated output directories
- parallel execution capability



# 7. Hyperparameter Tuning Strategy

Optimization followed a staged approach:

1. Structural model selection (Phase 1)
2. Parameter sensitivity analysis (Phase 2)
3. Integration testing (Phase 3)

This prevents parameter confounding and enables interpretable comparisons.



# 8. Comparative Analysis

Models were compared across:

- GRU vs LSTM architectures
- feature set complexity
- temporal configurations
- model capacity scaling
- optimization parameters

All comparisons used persistence‑relative skill scores.

Results show improvements primarily originate from correct temporal representation rather than increasing architectural complexity.



# 9. Experimental Progression

The final model was obtained through a structured multi-phase experimental process.
Each phase isolates a different aspect of the learning problem and evaluates configurations relative to the persistence baseline.



## Phase 1 — Structural Model Selection

### Objective
Identify suitable model components before numerical tuning.

### Parameters Tested

| Category | Values |
|---|---|
| Feature Sets | F0 (ratios), F1 (ratios + motion), F2 (ratios + motion + hydrometeor stats) |
| Architecture | GRU, LSTM |
| Loss Function | MSE, Huber |

### Number of Experiments
3 feature sets × 2 architectures × 2 losses = 12 runs

### Best Configuration
- Architecture: **GRU**
- Features: **F1 (motion enabled)**
- Loss: **MSE**

### Best Run Skill (vs Persistence)

| Target | Skill |
|---|---|
| Rain rate | 0.131 |
| Cloud base | 0.066 |
| TQC | -0.067 |
| TQI | -0.105 |

### Key Finding
Motion features improve predictive skill, while additional hydrometeor statistics do not provide measurable benefit. GRU models outperform LSTM with lower computational cost.

## Phase 2 — Parameter Sensitivity Analysis

### Objective

Phase 2 investigates which aspects of the model configuration control predictive performance.
Instead of performing a single large grid search, parameters were explored in three independent sweeps:

- model capacity,
- temporal representation,
- optimization dynamics.

Each sweep modifies only one parameter group while keeping all others fixed to the Phase-2 reference configuration.

---

### Sweep A — Model Capacity

#### Objective
Determine whether predictive performance is limited by model size or network depth.

#### Parameters Tested

| Parameter | Values |
|---|---|
| RNN Units | 32, 64, 128, 256 |
| Number of Layers | 1, 2, 3, 4 |

#### Number of Experiments

4 units × 4 layers = 16 runs


#### Best Configuration
- Units: **128**
- Layers: **1**
- Architecture: GRU

#### Best Run Skill (vs Persistence)

| Target | Skill |
|---|---|
| Rain rate | ~0.13 |
| Cloud base | ~0.07 |
| TQC | negative |
| TQI | negative |

#### Key Finding
Increasing depth does not improve performance. Moderate model width is sufficient, indicating that the problem is not capacity-limited.

---

### Sweep B — Temporal Representation

#### Objective
Identify the temporal scale at which cloud evolution becomes predictable.

#### Parameters Tested

| Parameter | Values |
|---|---|
| Input Window | 20, 30, 60, 90 minutes |
| Forecast Horizon | 5, 10, 20 minutes |
| Sampling Mode | overlapping / non-overlapping outputs |

#### Number of Experiments

4 input windows × 3 horizons × 2 sampling modes = 24 runs


#### Best Configuration
- Input window: **20–30 minutes**
- Forecast horizon: **20 minutes**
- Overlapping samples

#### Best Run Skill (vs Persistence)

| Target | Skill |
|---|---|
| Rain rate | 0.167 |
| Cloud base | 0.108 |
| TQC | ~0.03 |
| TQI | ~0.02 |

#### Key Finding
Temporal alignment dominates predictive performance. Forecast skill peaks when input history and forecast horizon match the intrinsic cloud evolution timescale (~20 minutes).

---

### Sweep C — Optimization Dynamics

#### Objective
Evaluate sensitivity of model performance to training hyperparameters and regularization.

#### Parameters Tested

| Parameter | Values |
|---|---|
| Learning Rate | 5e-4, 1e-3, 2e-3 |
| Dropout | 0.1, 0.2, 0.3 |
| Batch Size | 32, 64, 128 |

#### Number of Experiments

3 × 3 × 3 = 27 runs


#### Best Configuration
- Learning rate: **0.002**
- Dropout: **0.1**
- Batch size: **32**

#### Best Run Skill (vs Persistence)

| Target | Skill |
|---|---|
| Rain rate | ~0.16 |
| Cloud base | ~0.10 |
| TQC | slightly positive |
| TQI | slightly positive |

#### Key Finding
Optimization parameters influence training stability but produce smaller performance changes compared to temporal configuration.

---

### Phase 2 Summary

Across all sweeps:

- Temporal configuration produces the largest performance gains.
- Model capacity has secondary importance.
- Optimization settings fine-tune stability rather than predictive capability.


## Phase 3 — Integration of Best Components

### Objective
Test whether individually optimal settings remain effective when combined.

### Parameters Tested

| Parameter | Values |
|---|---|
| RNN Units | 128, 256 |
| Layers | 1 |
| Input Window | 20, 30 min |
| Forecast Horizon | 20 min |
| Random Seeds | 2 per configuration |

### Number of Experiments
2 units × 2 input windows × 2 seeds = 8 runs

### Best Configuration
- Architecture: **GRU**
- Units: **128**
- Layers: **1**
- Input window: **20 minutes**
- Horizon: **20 minutes**

### Best Run Skill (vs Persistence)

| Target | Skill |
|---|---|
| Rain rate | 0.175 |
| Cloud base | 0.175 |
| TQC | 0.10 |
| TQI | 0.09 |

### Key Finding
Performance saturates with moderate model capacity. Improvements arise primarily from temporal representation rather than increased complexity.


# 10. Final Model (Full Dataset)

Configuration:

- GRU
- 1 layer
- 128 units
- input 20 min
- horizon 20 min
- stride 10 min
- learning rate 0.002
- dropout 0.1
- batch size 32
- datashare 1.0

Performance:

| Target | Skill |
|---|---|
| Rain | 0.218 |
| Cloud base | 0.161 |
| TQC | 0.144 |
| TQI | 0.082 |
| Mean | 0.151 |



# 11. Conclusions

Key findings:

- Short‑term cloud evolution is partially predictable.
- Models cannot reproduce fine‑scale dynamics.
- Mean tendencies, especially rain formation, are captured.
- Motion features provide strongest predictive signal.
- Effective predictability timescale ≈ 20 minutes.

The final configuration serves as the reference model for evaluation and comparison.
