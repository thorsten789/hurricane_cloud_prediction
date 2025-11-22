# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: [Probabilistic spatiotemporal seasonal sea ice presence forecasting using sequence-to-sequence learning and ERA5 data in the Hudson Bay region]

  - **[Link]**[https://tc.copernicus.org/articles/16/3753/2022/#section2]
  - **Objective**: Seasonal prediction of sea ice coverage probability using ERA5 data and sequence-to-sequence learning models.
  - **Methods**: Encoder-decoder architecture with ConvLSTM in the encoder, followed by RNN decoder, plus a time-distributed network-in-network to maintain spatial resolution. Uses sequence-to-sequence to predict probabilities over future days.
  - **Outcomes**: The model is capable of making probabilistic predictions for sea ice patterns over months. It shows good performance compared to baselines.
  - **Relation to the Project**:Prediction over multiple time steps (nf) based on historical time series (nh). The use of ConvLSTM + Seq2Seq is directly transferable—especially if you want to learn how to efficiently model spatiotemporal (here vertical-temporal) relationships.

- **Source 2**: [Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Model]

  - **[Link]**[https://elib.dlr.de/216616/1/aies-AIES-D-24-0096.1%20%282%29.pdf]
  - **Objective**:Objective: Investigation of how well vertical profiles from NWP/data can be used with ML to predict thunderstorms/convective events (and associated intensity).
  - **Methods**:Methods: ML classifiers (e.g., gradient boosting/neural networks) on vertical profile sets; feature interpretation for physical consistency (e.g., ice content near tropopause).
  - **Outcomes**:Good performance in thunderstorm prediction; models use physically plausible patterns (ice/moisture signatures, instability measures).
  - **Relation to the Project**:explicitly uses vertical profiles to predict convective development/thunderstorms. Helpful for feature selection (which levels, how to normalize) and for interpretable ML methods.

- **Source 3**: [STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting]

  - **[https://arxiv.org/pdf/1912.00134]**
  - **Objective**: To develop a deep learning model that captures both spatial and temporal correlations in meteorological data in order to predict future weather conditions.
  - **Methods**: Sequence-to-sequence network consisting solely of convolutional layers (without RNN) to model spatiotemporal data, with encoder-decoder architecture.
  - **Outcomes**: The model exceeds or matches the performance of RNN-based approaches and is significantly faster in training. For temperature and precipitation predictions, it achieved better accuracy and faster training times than an RNN reference.
  - **Relation to the Project**: Our project also has temporal sequences (time steps) plus vertical profiles (spatial in the vertical), i.e., a spatio-temporal structure. A pure CNN-Seq2Seq such as STConvS2S could be a good starting point for learning such dependencies—especially since it enables the prediction of future time steps (nf) based on past ones (nh) without RNN overhead.

- **Source 4**: [Real-World Machine Learning]

  -**Link**: Physical book by Brink et al. (ISBN: 9781617291920)
  - **Objective**: A practical machine learning handbook that explains the complete lifecycle of an ML project in a real-world environment—from problem definition to data preparation, feature engineering, modeling, evaluation, and deployment. The book focuses in particular on how to implement ML with real, flawed, complex data.
  - **Methods**: The book covers various methods, not in terms of mathematical depth, but as an “engineering manual”:
    - Data transformation, data quality, exploratory data analysis (EDA)
    - Feature engineering, especially for non-image data
    - Classic ML models (GBM, random forest, GLM), first steps with neural networks
    - Pipeline design, model validation, train/test splits
    - Dealing with imbalance, rare events, outliers
    - Model interpretability
  - **Outcomes**: The book teaches best practices that are crucial for any ML project:
    - How to create robust, reproducible datasets
    - Choosing the right metrics
    - Dealing with real-world challenges: incomplete data, noise, data volume, scaling
    - How to interpret models and perform debugging
    - Focus on iterative improvement instead of “one-shot training”
  - **Relation to the Project**: Although the book is not meteorological or DL-specific, it is extremely relevant to the project—because we are facing a “real” data science project with complex raw data, different feature types, and many design decisions. The book supports you in these areas in particular:
  - Feature engineering
    - The profiles can be improved with derived features (integrals, layer means, gradients, etc.).
    - The book systematically explains how to identify and evaluate such features.
  - Validation strategy (crucial for time series)
    - It warns against random train/test splits, which would be incorrect for time series.
    - Cell-based split
    - Rolling windows
    - Temporal CV → directly applicable.
  - Model comparison and selection
    - Although DL is important, classic models (GBM, Random Forest, Linear Models) should be used as a baseline → the book explicitly advises this.
    - For the project: Baselines such as “Persistence,” “Statistics on profile” + GBM are very valuable.
  - Dealing with rare events
    - Heavy rain is a “rare event” → the book explains how to deal with imbalance, weighted losses, and sampling.
    - Very useful for rain intensity prediction.
  - Interpretability using Shapley values, feature importance, residual analysis.
