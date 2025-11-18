# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: [tobac: Tracking and Object-Based Analysis of Clouds]

  - **[Link]**[https://gmd.copernicus.org/preprints/gmd-2019-105/gmd-2019-105-manuscript-version4.pdf]
  - **Objective**:Objective: Presentation of TOBAC as a flexible framework for the identification, tracking, and object-based analysis of clouds in 2D/3D datasets (model/satellite data)
  - **Methods**: Algorithmic object definition (thresholds, morphology), tracking over time steps, calculation of object metrics (size, lifetime, splits/mergers), modularly implemented in Python.
  - **Outcomes**: Open-source package with sample workflows; describes how parent–child relationships (splits/mergers) and temporally linked object attributes are created.
  - **Relation to the Project**:TOBAC provides exactly the object/cell structure, parent–child information, and time series that your dataset has; useful for preprocessing, feature definition (e.g., life stage, split/merge flags), and reproducibility

- **Source 2**: [Inferring Thunderstorm Occurrence from Vertical Profiles of Convection-Permitting Simulations: Physical Insights from a Physical Deep Learning Model]

  - **[Link]**[https://elib.dlr.de/216616/1/aies-AIES-D-24-0096.1%20%282%29.pdf]
  - **Objective**:Objective: Investigation of how well vertical profiles from NWP/data can be used with ML to predict thunderstorms/convective events (and associated intensity).
  - **Methods**:Methods: ML classifiers (e.g., gradient boosting/neural networks) on vertical profile sets; feature interpretation for physical consistency (e.g., ice content near tropopause).
  - **Outcomes**:Good performance in thunderstorm prediction; models use physically plausible patterns (ice/moisture signatures, instability measures).
  - **Relation to the Project**:explicitly uses vertical profiles to predict convective development/thunderstorms. Helpful for feature selection (which levels, how to normalize) and for interpretable ML methods.

- **Source 3**: [Title of Source 3]

  - **[Link]**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:
