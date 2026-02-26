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

-   datashare = 0.2
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


