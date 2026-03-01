# Analysis and Prediction of Cloud Development in Hurricane Systems

## Repository Link

<https://github.com/thorsten789/hurricane_cloud_prediction>

## Description

This project analyzes the development of clouds within simulated hurricane systems using machine learning methods.

The dataset is based on high-resolution simulations of **Hurricane Paulette (2020)** produced with the ICON weather model. Cloud objects were identified and tracked using the **TOBAC (Tracking and Object-Based Analysis of Clouds)** framework via the COMIN interface.

Each tracked cloud object contains:

- temporal evolution of cloud properties
- parent–child relations (splitting and merging)
- vertical profiles of thermodynamic and microphysical variables
- derived scalar diagnostics (e.g., rain rate, cloud water, cloud ice)

The main objective is to investigate **whether short-term cloud evolution can be predicted from the physical state and recent history of a cloud system**.



## Task Type

Time-series regression / multi-target forecasting



## Results Summary

### Best Model Performance

- **Best Model:** GRU-based multi-task recurrent neural network
- **Evaluation Metric:** Persistence-relative RMSE skill score
- **Forecast Setup:** 20-minute prediction horizon

| Target | Skill vs Persistence |
|---|---|
| Rain rate | **0.218** |
| Cloud base height | **0.161** |
| Total cloud water (TQC) | **0.144** |
| Total cloud ice (TQI) | **0.082** |

Skill is defined as:

```
Skill = 1 − RMSE_model / RMSE_persistence
```

where persistence assumes the future state equals the current observation.



### Model Comparison

- **Baseline:** Persistence forecast (no cloud evolution assumed)
- **Improvement:** Clear improvement for rain prediction and moderate improvement for structural cloud properties.
- **Architecture comparison:** GRU models consistently outperformed LSTM models.
- **Feature comparison:** Motion features improved predictions; additional hydrometeor statistics did not improve performance.



### Key Insights

- Short-term cloud evolution is only partially predictable.
- The model **cannot reproduce detailed small-scale dynamics** of individual clouds.
- However, it successfully captures **average tendencies**, especially for rain formation.
- Motion information provides stronger predictive signal than detailed microphysical statistics.
- Increasing model complexity beyond moderate size does not improve performance.

An effective predictability timescale of roughly **20 minutes** emerged from the experiments.

**To become practically applicable, the model would need to capture variability beyond mean tendencies. Achieving reliable prediction of fine-scale dynamics remains an important direction for future work.**

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/README.md)**
3. **[Baseline Model](2_BaselineModel/README.md)**
4. **[Model Definition and Evaluation](3_Model/README.md)**
5. **[Presentation](4_Presentation/README.md)**




## Cover Image

![Project Cover Image](CoverImage/cover_image.png)
