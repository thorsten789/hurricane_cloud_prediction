# Cloud Multi-Task Forecasting Framework

## Technical Documentation -- Phase 0 & Phase 1

------------------------------------------------------------------------

# 1. Overview

`cloud_multitask_train.py` is a configurable multi-task deep learning
training framework designed to forecast the future evolution of tracked
cloud systems.

The model predicts multiple physically meaningful cloud properties
simultaneously using time-series inputs derived from:

-   Scalar cloud diagnostics
-   Vertical microphysical profiles
-   Engineered physical features
-   Optional motion dynamics (derived from lon/lat)

The framework is designed for large-scale HPC experimentation (SLURM job
arrays), reproducibility, and structured scientific evaluation.

------------------------------------------------------------------------

# 2. Model Philosophy

The goal is not merely to predict rain rate, but to forecast cloud
evolution in a physically interpretable way.

The system predicts:

-   Surface rain rate
-   Cloud base height
-   Total cloud liquid water (tqc)
-   Total cloud ice (tqi)

All targets are predicted as future sequences (multi-step forecasting).

This enables:

-   Horizon-wise evaluation
-   Skill score comparison against persistence baseline
-   Multi-task physical consistency

------------------------------------------------------------------------

# 3. Input Feature System

## 3.1 Scalar Features

Used scalars:

-   cape_ml_L00
-   cin_ml_L00
-   rain_gsp_rate_L00
-   tqc_L00
-   tqi_L00
-   area_m2
-   remaining_lifetime (label only, excluded from input)

Optional engineered scalar ratios:

-   tqc / tqi
-   tqc / rainrate (rain efficiency)

------------------------------------------------------------------------

## 3.2 Profile-Based Features

Vertical profiles are extracted for:

-   qc
-   qi
-   qr
-   qg
-   qs
-   qv
-   roh
-   w

Engineered profile features include:

-   Cloud base
-   Cloud top
-   Cloud thickness
-   Cloud mass
-   Rain mass
-   Mean vertical velocity
-   Max vertical velocity
-   Height of max qc
-   Height of max w
-   Center of mass
-   Std vertical velocity

Optional additional hydrometeor statistics:

-   max_qx
-   height_max_qx
-   sum_qx

------------------------------------------------------------------------

## 3.3 Motion Features (Optional)

Derived from lon/lat using relative displacement:

-   dx, dy
-   speed
-   heading (sin/cos encoding)
-   acceleration
-   turn rate

Absolute geographic position is NOT used as input.

------------------------------------------------------------------------

# 4. Model Architecture

Shared encoder:

Input (T_in, N_features) → RNN layers (LSTM or GRU) → Layer
Normalization → Global Average Pooling → Global Max Pooling → Dense
layer → Dropout

Multi-task output heads:

For each target: Dense(N_output_steps)

All tasks predict full future sequences.

------------------------------------------------------------------------

# 5. Training & Evaluation

Training features:

-   Early stopping
-   ReduceLROnPlateau
-   Model checkpointing
-   StandardScaler for X and optional Y scaling

Evaluation metrics:

-   MAE
-   RMSE
-   Bias
-   Horizon-wise error curves
-   Persistence baseline comparison
-   Skill score calculation

All outputs are stored:

-   model files
-   config snapshot
-   scalers
-   predictions
-   metrics.json
-   plots

------------------------------------------------------------------------

# 6. HPC Design Considerations

The script is designed for SLURM job arrays.

Key properties:

-   Fully CLI configurable
-   Unique run-name per job
-   No shared-state conflicts
-   Reproducible configuration snapshot
-   Suitable for large-scale sweeps

------------------------------------------------------------------------

# 7. Phase 1 Experimental Design

## Objective

Phase 1 determines which structural model components and feature groups
provide meaningful predictive signal before hyperparameter tuning.

This is a structured ablation study.

------------------------------------------------------------------------

## Fixed Parameters

-   datashare = 0.5
-   epochs = 20
-   input window = 30 minutes
-   forecast horizon = 10 minutes
-   stride = 10 minutes
-   rnn_units = 64
-   n_layers = 2
-   dropout = 0.2
-   batch_size = 64
-   targets = rain, cloud_base, tqc, tqi

------------------------------------------------------------------------

## Variables Tested

### Feature Complexity (3)

F0 -- Scalar ratios only\
F1 -- Ratios + Motion\
F2 -- Ratios + Motion + Hydrometeor profile stats

Purpose: Determine contribution of motion dynamics and detailed
microphysics.

------------------------------------------------------------------------

### RNN Type (2)

-   LSTM
-   GRU

Purpose: Compare capacity versus efficiency.

------------------------------------------------------------------------

### Loss Function (2)

-   MSE
-   Huber

Purpose: Evaluate robustness against heavy-tailed rain rate
distributions.

------------------------------------------------------------------------

## Total Phase 1 Runs

3 feature sets × 2 RNN types × 2 loss functions\
= 12 experiments

Executed via SLURM array.



# 8. Phase 1 Results and Conclusions

## Persistence Baseline Model

To evaluate whether the neural network learns meaningful predictive dynamics, all results are compared against a **persistence baseline model**.

The persistence model assumes that the future state of the system remains equal to its most recent observed state. For a forecast horizon Δt, the prediction is defined as:

    y_hat(t + Δt) = y(t)

In other words, the model predicts **no evolution** of the cloud system.

### Why Persistence is Used

Persistence is a strong and physically meaningful baseline for short-term atmospheric forecasting because many atmospheric variables evolve slowly relative to short prediction horizons. A model that cannot outperform persistence does not provide additional predictive information beyond the current observation.

Using persistence allows us to answer a fundamental question:

> Does the model learn dynamical evolution, or does it merely reproduce the current state?

## Evaluation Methodology

Model performance was evaluated per target using a persistence-relative skill score:

```
Skill = 1 - RMSE_model / RMSE_persistence
```

This metric is dimensionless and allows comparison across targets with different physical units.

Targets evaluated:

- Rain rate (`rain_gsp_rate_L00`)
- Cloud base height (`cloud_base`)
- Total cloud liquid water (`tqc_L00`)
- Total cloud ice (`tqi_L00`)

---

## Overall Predictive Skill

Mean skill averaged across all Phase 1 runs:

| Target | Mean Skill |
|---|---|
| Rain rate | **0.118** |
| Cloud base | **0.053** |
| TQC | **-0.092** |
| TQI | **-0.123** |

### Interpretation

- Rain prediction shows clear improvement over persistence.
- Cloud base exhibits weak but positive predictability.
- TQC and TQI remain persistence dominated at the investigated forecast horizon.

---

## Feature Set Analysis

Average skill by feature configuration:

| Feature Set | Rain Skill | Cloud Base Skill | TQC Skill | TQI Skill |
|---|---|---|---|---|
| F0 — Ratios | 0.120 | 0.053 | -0.093 | -0.089 |
| F1 — Motion | **0.121** | **0.063** | -0.091 | -0.148 |
| F2 — Motion + Hydro | 0.113 | 0.044 | -0.093 | -0.131 |

### Result

Motion features provide the strongest overall improvement, particularly for precipitation prediction.
Hydrometeor statistics do not improve skill under the tested configuration.

---

## Architecture Comparison

Average skill by recurrent architecture:

| Architecture | Rain Skill | Cloud Base Skill | TQC Skill | TQI Skill |
|---|---|---|---|---|
| **GRU** | **0.131** | **0.066** | -0.067 | -0.105 |
| LSTM | 0.105 | 0.041 | -0.118 | -0.141 |

### Result

GRU consistently achieves higher predictive skill while being computationally more efficient.

---

## Loss Function Comparison

| Loss | Rain Skill | Cloud Base Skill | TQC Skill | TQI Skill |
|---|---|---|---|---|
| Huber | 0.119 | 0.048 | -0.102 | -0.124 |
| MSE | 0.117 | 0.058 | -0.083 | -0.122 |

### Result

Both losses perform similarly. No systematic advantage was observed.

---

## Physical Interpretation

The experiments indicate two predictability regimes:

| Regime | Behaviour |
|---|---|
| Dynamical | Rain and cloud structure contain learnable temporal signal |
| Microphysical | Bulk hydrometeor quantities behave close to persistence |

Motion improves prediction of advective processes, whereas detailed hydrometeor statistics do not add independent predictive information at the studied time scale.

---

## Phase 2 Configuration

Based on Phase 1 results, the following configuration is fixed:

| Component | Selected Choice |
|---|---|
| Architecture | **GRU** |
| Feature Set | **F1 — Motion enabled** |
| Hydrometeor statistics | Disabled |
| Loss | MSE |
| Targets | rain, cloud_base, tqc, tqi |

---

## Phase 2 Focus

With structural choices fixed, Phase 2 investigates:

- Model capacity scaling
- Temporal window length
- Forecast horizon sensitivity
- Training dynamics (learning rate, dropout)

---

## Summary

Phase 1 establishes a stable and physically interpretable baseline:

- Rain prediction improves over persistence by ~11.79%.
- Motion information provides measurable predictive value.
- GRU architectures outperform LSTM while remaining computationally efficient.
- Increasing feature complexity alone does not guarantee improved predictive skill.

These results define the validated starting point for Phase 2 optimization.


# Phase 2 — Systematic Parameter Exploration



## Objective of Phase 2

Phase 2 aims to explore the parameter space surrounding the validated reference model in a controlled and interpretable manner.

Unlike Phase 1, which identified *structural model choices*, Phase 2 investigates **quantitative scaling behaviour** of the model along three independent dimensions:

1. Model capacity
2. Temporal representation
3. Optimization dynamics

Each dimension is evaluated independently against the Phase‑2 Reference Model.

The goal is **not yet to determine a final optimal configuration**, but to understand sensitivities and identify promising regions of parameter space for later combination testing (Phase 3).


---

## Experimental Design Philosophy

Phase 2 follows a *controlled one‑factor exploration* strategy:

- Only one parameter category is varied at a time.
- All remaining parameters remain fixed to the reference configuration.
- Results are always interpreted relative to the persistence‑skill evaluation framework established in Phase 1.

This approach ensures:

- interpretability of observed performance changes,
- avoidance of parameter confounding,
- efficient use of computational resources,
- reproducible experimental reasoning.



---

## Phase 2 Baseline Configuration

Based on the Phase 1 ablation study, a single reference configuration is selected as the **baseline model for all Phase 2 experiments**.

This configuration represents the best trade-off between predictive skill, physical interpretability, and computational efficiency.

| Component | Selected Baseline |
|---|---|
| Feature Set | **F1 — Ratios + Motion features** |
| Architecture | **GRU** |
| Loss Function | **MSE** |
| Hydrometeor statistics | Disabled |
| RNN Units | 64 |
| Layers | 2 |
| Dropout | 0.2 |
| Input Window | 30 min |
| Forecast Horizon | 10 min |
| Batch Size | 64 |

All Phase 2 experiments modify **exactly one parameter group at a time** relative to this baseline in order to isolate causal effects on predictive performance.

This configuration is hereafter referred to as:

> **Phase‑2 Reference Model**


---

## Sweep A — Capacity Scaling

### Objective

Determine whether predictive performance is limited by model representational capacity.

### Parameters

| Parameter | Values |
|---|---|
| RNN Units | 32, 64, 128, 256 |
| Number of Layers | 1, 2, 3, 4 |

Total experiments:

```
4 units × 4 layers = 16 runs
```

### Fixed Parameters

- Temporal configuration = reference model
- Optimization parameters = reference model

### Scientific Question

> Does increasing model capacity improve the representation of cloud dynamics?



---

## Sweep B — Temporal Representation

### Objective

Investigate how temporal context and sampling density influence forecast skill.

### Parameters

| Parameter | Values |
|---|---|
| Input Window | 20, 30, 60, 90 minutes |
| Forecast Horizon | 5, 10, 20 minutes |
| Sampling Mode | overlapping outputs / non‑overlapping outputs |

Sampling modes:

- **Overlapping outputs:** stride < forecast horizon  
  (denser temporal sampling)
- **Non‑overlapping outputs:** stride = forecast horizon  
  (independent forecast samples)

Total experiments:

```
4 input windows × 3 horizons × 2 stride modes = 24 runs
```

### Fixed Parameters

- Model capacity = reference model
- Optimization parameters = reference model

### Scientific Questions

- How much historical context is required for prediction?
- How does forecast skill degrade with horizon length?
- Does dense temporal sampling improve learning stability?



---

## Sweep C — Optimization Dynamics

### Objective

Assess sensitivity of training performance to optimization and regularization choices.

### Parameters

| Parameter | Values |
|---|---|
| Learning Rate | 5e‑4, 1e‑3, 2e‑3 |
| Dropout | 0.1, 0.2, 0.3 |
| Batch Size | 32, 64, 128 |

Total experiments:

```
3 × 3 × 3 = 27 runs
```

### Fixed Parameters

- Architecture = reference model
- Temporal setup = reference model

### Scientific Questions

- Is performance limited by optimization stability?
- Does additional regularization improve generalization?
- How sensitive is training to gradient noise scale?



---

## Phase 2 Workload Summary

| Sweep | Purpose | Runs |
|---|---|---|
| Sweep A | Capacity scaling | 16 |
| Sweep B | Temporal representation | 24 |
| Sweep C | Optimization dynamics | 27 |
| **Total** |  | **67 runs** |

All experiments are executed as independent SLURM array jobs using identical evaluation procedures.


