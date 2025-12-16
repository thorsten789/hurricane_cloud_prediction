# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection

- **Baseline Model Type:** Reduce the input sequence to single values (mean, min/max, gradients) and feed to Random Forest.
- **Rationale:** A Baseline Model should start with something simple. A Random Forest doesn't take long to train, yet already carries a lot of information.

### Model Performance

- **Evaluation Metric:** MAE 46.86 seconds | RMSE 139.39 seconds

### Evaluation Methodology

- **Data Split:** Train/Validation/Test split ratios: 70/15/15
- **Evaluation Metrics:** MAE and RMSE both produce human-understandable results. A fundamental problem in our Dataset Characteristics is the imbalance resulting from the prevalence of cloud cells with a very short lifetime. Because it punishes large errors stronger, we suppose RMSE to suit better to the task.

### Metric Practical Relevance

We feed the model with data in timesteps of 30 seconds. Thus the results of both the Mean Absolute Error and Root Mean Square Error metrics lie within the scope of few timesteps, which is very good.

## Next Steps

This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
