# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

# Dataset Information

## Dataset Source
- **Dataset Link:**  
  *This dataset is part of ongoing, unpublished research. It is therefore not publicly accessible.*  
  Access may be granted upon request.

  A Data Sample is included in this Repo. 

- **Dataset Owner/Contact:**  
  Dr. Matthias Faust

  Leibniz Institute for Tropospheric Research (TROPOS)  
  *Modelling Department*  
  Contact: [faust@tropos.de]

## Dataset Characteristics

### Scope and Structure
The dataset contains **4,500 individual cloud tracks (cloud cells)** extracted from a high-resolution ICON simulation of Hurricane Paulette (7–8 September 2020).  
Each cloud track is represented as a **time series** with a temporal resolution of **30 seconds**. Track lengths vary strongly due to natural differences in cloud lifetime.

Each time step contains:
- metadata (time, frame, lat/lon, cell id)  
- scalar physical variables (e.g., LWP, IWP, CAPE, CIN, rain rate, area)  
- full vertical profiles of microphysical and dynamical variables on ~50 model levels (hydrometeors, water vapor, air density, vertical velocity)

### Dataset Size
- **Number of cloud cells:** 4,500  
- **Temporal resolution:** 30 s  
- **Features per time step:** several hundred (dominated by vertical profiles)  
- **Track lengths:** from a few time steps up to several hours

## Prediction Task (Target)

### Goal
Given an **incomplete cloud track up to time t**, the model predicts:
1. **the remaining lifetime** of the cloud  
2. **the future rain rate** over the next steps (rain-rate sequence)

This corresponds to a **joint regression + sequence forecasting task**:

| Component | Type | Description |
|----------|------|-------------|
| **Remaining lifetime** | Regression | `lifetime_remaining_s = t_end − t_current` |
| **Future rain-rate sequence** | Seq2Seq regression | Predict `rain(t+1 … t+k)` |

The input is a multivariate, irregular-length cloud time series;  
the output is a predicted continuation of the sequence.

## Label Description

### 1. Remaining Lifetime
- **Label name:** `remaining_lifetime_s`  
- **Type:** regression  
- **Description:** Time until cloud dissipates, measured from the last input time step  
- **Distribution:** Strongly right-skewed; most clouds die quickly

### 2. Future Rain-Rate Sequence
- **Label name:** `rain_rate_future[t+1 ... t+k]`  
- **Type:** sequence regression  
- **Description:** Predict the short-term evolution of grid-scale precipitation rate  
- **Distribution:** Highly sparse and right-skewed (rain occurs only in a minority of time steps)

## Feature Description

### 1. Scalar Features
- Latitude, longitude  
- Area (m²)  
- LWP, IWP, TQC, TQI  
- CAPE, CIN  
- Precipitation rate  

### 2. Vertical Profile Features (~50 levels each)
- Hydrometeor mixing ratios: qc, qi, qs, qg, qr  
- Water vapor (qv)  
- Air density (roh)  
- Vertical velocity (w)

These profile features describe the full vertical structure of clouds and convection, essential for forecasting cloud development and precipitation.

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [Exploratory Data Analysis](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
