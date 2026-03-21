# Calgary solar energy production forecaster

## Problem statement

Calgary has committed to sourcing increasing municipal energy from solar PV installations, but production varies dramatically with season and facility. Accurate forecasting of monthly solar output is critical for grid planning, budgeting, and tracking sustainability targets. This project builds a forecasting system for each municipal solar installation using 2,300+ monthly production records.

## Approach

- Fetched solar PV production and site data from Calgary Open Data via Socrata API
- Engineered cyclical month encoding, rolling averages (3/6/12-month), and lag features per facility
- Trained Ridge Regression, Random Forest, and XGBoost regressors
- Used temporal train/test split with MAE, RMSE, R-squared, and MAPE evaluation
- Implemented iterative multi-step forecasting for future months

## Key results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| R-squared | ~0.86 |
| MAPE | ~20% |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_08_solar_energy_forecaster/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```
