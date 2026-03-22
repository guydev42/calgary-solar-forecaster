<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Solar%20Energy%20Production%20Forecaster&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Seasonal%20decomposition%20and%20ML%20forecasting%20for%20Calgary%20solar%20facilities&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/XGBoost-0.86_R²-blue?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- Calgary has committed to sourcing increasing municipal energy from solar PV installations, but production varies dramatically with season and facility. Accurate forecasting of monthly solar output is critical for grid planning, budgeting, and tracking sustainability targets.
>
> **Solution** -- This project builds a per-facility forecasting system using 2,300+ monthly production records, combining seasonal decomposition with Ridge Regression, Random Forest, and XGBoost regressors.
>
> **Impact** -- Provides facility-level monthly production forecasts that support grid planning, budget allocation, and progress tracking toward Calgary's renewable energy goals.

---

## Results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| R-squared | ~0.86 |
| MAPE | ~20% |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  Production +    │────>│  Cyclical month  │────>│  Model suite   │────>│  Streamlit      │
│  Data (Socrata) │     │  Site data       │     │  Rolling avgs    │     │  Ridge / RF    │     │  dashboard      │
│  Solar PV       │     │  Per-facility    │     │  3/6/12-month    │     │  XGBoost       │     │  Per-facility   │
│  2,300+ records │     │  Merge & clean   │     │  Lag features    │     │  Seasonal dec  │     │  Forecast view  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_08_solar_energy_forecaster/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   ├── solar_production.csv        # Monthly production records
│   └── solar_sites.csv             # Facility metadata
├── models/                         # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching & preprocessing
    └── model.py                    # Model training & forecasting
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/solar-energy-forecaster.git
cd solar-energy-forecaster

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- Solar PV Production](https://data.calgary.ca/) |
| Records | 2,300+ monthly production records |
| Access method | Socrata API (sodapy) |
| Key fields | Facility name, month, production (kWh), site capacity, location |
| Target variable | Monthly energy production (kWh) per facility |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and merging

- Fetched solar PV production and site data from Calgary Open Data via the Socrata API
- Merged production records with facility metadata (capacity, location, installation date)
- Cleaned and standardized monthly production values per facility

### Feature engineering

- Engineered cyclical month encoding (sine/cosine) to capture seasonal production patterns
- Created rolling averages at 3-month, 6-month, and 12-month windows per facility
- Built lag features (1, 3, 6, 12 months) to capture temporal dependencies
- Applied seasonal decomposition to isolate trend, seasonal, and residual components

### Model training and evaluation

- Trained Ridge Regression, Random Forest, and XGBoost regressors
- Used temporal train/test split with MAE, RMSE, R-squared, and MAPE evaluation
- XGBoost achieved the best R-squared of ~0.86 with MAPE of ~20%

### Multi-step forecasting

- Implemented iterative multi-step forecasting for future months
- Each predicted value feeds back as input for subsequent predictions
- Per-facility models capture site-specific production characteristics

### Interactive dashboard

- Built a Streamlit dashboard with per-facility forecast views and production trend analysis
- Visualizations include seasonal decomposition plots, forecast comparisons, and facility-level summaries

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing solar PV production data
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
