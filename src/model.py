"""
Model module for Calgary Solar Energy Production Forecaster.

Implements time series regression models for forecasting monthly
solar PV production across Calgary municipal facilities.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Feature columns used by all models
FEATURE_COLUMNS = [
    "month_sin",
    "month_cos",
    "year",
    "lag_1m",
    "lag_3m",
    "lag_12m",
    "rolling_avg_3m",
    "rolling_avg_6m",
    "rolling_avg_12m",
]


def get_models() -> dict:
    """
    Return a dictionary of model instances for training.

    Returns
    -------
    dict
        Mapping of model name to scikit-learn compatible estimator.
    """
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    return models


def temporal_train_test_split(
    df: pd.DataFrame,
    test_months: int = 12,
) -> tuple:
    """
    Split data into train and test sets using a temporal cutoff.

    The most recent ``test_months`` months of data are held out for
    testing, preserving the time series ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered production data with ``period_dt`` column.
    test_months : int
        Number of most recent months to reserve for testing.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Training and testing DataFrames.
    """
    df = df.sort_values("period_dt").reset_index(drop=True)

    unique_periods = df["period_dt"].sort_values().unique()
    if len(unique_periods) <= test_months:
        cutoff_idx = len(unique_periods) // 2
    else:
        cutoff_idx = len(unique_periods) - test_months

    cutoff_date = unique_periods[cutoff_idx]

    train = df[df["period_dt"] < cutoff_date].copy()
    test = df[df["period_dt"] >= cutoff_date].copy()

    return train, test


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> tuple:
    """
    Extract feature matrix and target vector from the DataFrame.

    Drops rows with any NaN values in the feature or target columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing feature and target columns.
    feature_cols : list, optional
        Feature column names. Defaults to FEATURE_COLUMNS.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series, pd.DataFrame)
        Feature matrix X, target vector y, and the cleaned DataFrame.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    target_col = "solar_pv_production_kwh"
    required = feature_cols + [target_col]

    clean = df.dropna(subset=required).copy()
    X = clean[feature_cols]
    y = clean[target_col]

    return X, y, clean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, R2, and MAPE scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — guard against division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_and_evaluate(
    df: pd.DataFrame,
    test_months: int = 12,
) -> dict:
    """
    Train all models and evaluate on a temporal test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered production DataFrame.
    test_months : int
        Number of recent months for testing.

    Returns
    -------
    dict
        Dictionary containing:
        - 'results': list of dicts with model name, metrics, predictions
        - 'best_model_name': name of the best model by RMSE
        - 'best_model': fitted best model object
        - 'test_data': test DataFrame with predictions
    """
    train_df, test_df = temporal_train_test_split(df, test_months=test_months)

    X_train, y_train, train_clean = prepare_features(train_df)
    X_test, y_test, test_clean = prepare_features(test_df)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Insufficient data after cleaning. "
            f"Train size: {len(X_train)}, Test size: {len(X_test)}"
        )

    models = get_models()
    results = []
    best_rmse = float("inf")
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test.values, y_pred)
        results.append({
            "model_name": name,
            "metrics": metrics,
            "predictions": y_pred,
        })

        print(
            f"  {name}: MAE={metrics['MAE']:.1f}, "
            f"RMSE={metrics['RMSE']:.1f}, "
            f"R2={metrics['R2']:.4f}, "
            f"MAPE={metrics['MAPE']:.2f}%"
        )

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model_name = name
            best_model = model

    # Attach predictions from the best model to test data
    test_clean = test_clean.copy()
    best_preds = [
        r["predictions"] for r in results if r["model_name"] == best_model_name
    ][0]
    test_clean["predicted_kwh"] = best_preds

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "test_data": test_clean,
    }


def generate_forecast(
    model,
    df: pd.DataFrame,
    facility_name: str,
    n_months: int = 12,
) -> pd.DataFrame:
    """
    Generate future production forecasts for a specific facility.

    Uses an iterative approach: each forecasted month's prediction
    is fed back as a lag feature for subsequent months.

    Parameters
    ----------
    model : estimator
        A trained scikit-learn compatible model.
    df : pd.DataFrame
        Historical feature-engineered data.
    facility_name : str
        Name of the facility to forecast.
    n_months : int
        Number of future months to forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame with forecast dates and predicted production.
    """
    facility_data = (
        df[df["facility_name"] == facility_name]
        .sort_values("period_dt")
        .copy()
    )

    if len(facility_data) == 0:
        raise ValueError(f"No data found for facility: {facility_name}")

    last_date = facility_data["period_dt"].max()
    recent_values = facility_data["solar_pv_production_kwh"].values

    forecasts = []

    for i in range(n_months):
        forecast_date = last_date + pd.DateOffset(months=i + 1)
        month = forecast_date.month
        year = forecast_date.year

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Build lag and rolling features from available history
        all_values = np.concatenate([recent_values, [f["predicted_kwh"] for f in forecasts]])

        lag_1 = all_values[-1] if len(all_values) >= 1 else np.nan
        lag_3 = all_values[-3] if len(all_values) >= 3 else np.nan
        lag_12 = all_values[-12] if len(all_values) >= 12 else np.nan

        roll_3 = np.mean(all_values[-3:]) if len(all_values) >= 3 else np.mean(all_values)
        roll_6 = np.mean(all_values[-6:]) if len(all_values) >= 6 else np.mean(all_values)
        roll_12 = np.mean(all_values[-12:]) if len(all_values) >= 12 else np.mean(all_values)

        features = pd.DataFrame([{
            "month_sin": month_sin,
            "month_cos": month_cos,
            "year": year,
            "lag_1m": lag_1,
            "lag_3m": lag_3,
            "lag_12m": lag_12,
            "rolling_avg_3m": roll_3,
            "rolling_avg_6m": roll_6,
            "rolling_avg_12m": roll_12,
        }])

        prediction = model.predict(features)[0]
        prediction = max(0, prediction)  # Production cannot be negative

        forecasts.append({
            "period_dt": forecast_date,
            "year": year,
            "month": month,
            "facility_name": facility_name,
            "predicted_kwh": round(prediction, 2),
        })

    return pd.DataFrame(forecasts)


def save_model(model, model_name: str) -> str:
    """
    Save a trained model to the models directory using joblib.

    Parameters
    ----------
    model : estimator
        Trained model to persist.
    model_name : str
        Name identifier for the saved model file.

    Returns
    -------
    str
        File path where the model was saved.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return str(filepath)


def load_model(model_name: str):
    """
    Load a previously saved model from the models directory.

    Parameters
    ----------
    model_name : str
        Name identifier of the saved model file.

    Returns
    -------
    estimator
        The loaded model object.
    """
    filepath = MODELS_DIR / f"{model_name}.joblib"
    if not filepath.exists():
        raise FileNotFoundError(f"No saved model found at {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    from data_loader import load_and_prepare_data

    data = load_and_prepare_data()
    production = data["production"]

    print("=" * 60)
    print("Training and evaluating models...")
    print("=" * 60)

    evaluation = train_and_evaluate(production, test_months=12)

    print(f"\nBest model: {evaluation['best_model_name']}")

    # Save best model
    save_model(evaluation["best_model"], evaluation["best_model_name"])

    # Generate sample forecast
    facilities = production["facility_name"].unique()
    if len(facilities) > 0:
        sample_facility = facilities[0]
        print(f"\nForecasting 12 months for: {sample_facility}")
        forecast = generate_forecast(
            evaluation["best_model"], production, sample_facility, n_months=12
        )
        print(forecast[["period_dt", "predicted_kwh"]].to_string(index=False))
