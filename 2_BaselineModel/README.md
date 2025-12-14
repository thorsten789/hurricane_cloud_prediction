# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection

- **Baseline Model Type:** Reduce the input sequence to single values (mean, min/max, gradients) and feed to Random Forest.
- **Rationale:** A Baseline Model should start with something simple. A Random Forest doesn't take long to train, yet already carries a lot of information.

### Model Performance

- **Evaluation Metric:** MAE: 732.65 s; RMSE: 1227.20 s | MAE 46,86 s; RMSE 139,93 s
- [**Performance Score:** e.g., 85% accuracy, F1-score of 0.78, MSE of 0.15]
- [**Cross-Validation Score:** Mean and standard deviation of CV scores, e.g., 0.82 ± 0.03]

### Evaluation Methodology

- **Data Split:** Train/Validation/Test split ratios: 70/15/15
- **Evaluation Metrics:** MAE and RMSE produce human-understandable results. A fundamental problem in our Dataset Characteristics is the imbalance resulting from the prevalence of cloud cells with a very short lifetime. Therefore we suppose RMSE to suit better because of stronger punishment of large errors.

### Metric Practical Relevance

[Explain the practical relevance and business impact of each chosen evaluation metric. How do these metrics translate to real-world performance and decision-making? What do the metric values mean in the context of your specific problem domain?]

## Next Steps

This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
