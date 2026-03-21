"""
Data loader for Calgary Solar Energy Production Forecaster.

Fetches solar production and facility data from Calgary Open Data portal,
caches locally, and engineers features for time series forecasting.
"""

import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sodapy import Socrata

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Calgary Open Data endpoint
CALGARY_OPEN_DATA_DOMAIN = "data.calgary.ca"

# Dataset identifiers
SOLAR_PRODUCTION_DATASET = "iric-4rrc"
SOLAR_SITES_DATASET = "tbsv-89ps"

# Default data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_solar_production(
    limit: int = 50000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch solar PV production data from Calgary Open Data.

    Parameters
    ----------
    limit : int
        Maximum number of records to retrieve.
    force_refresh : bool
        If True, bypass the local cache and re-download.

    Returns
    -------
    pd.DataFrame
        Raw solar production data.
    """
    cache_path = DATA_DIR / "solar_production.csv"

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached solar production data from %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Fetching solar production data from Calgary Open Data...")
    try:
        client = Socrata(CALGARY_OPEN_DATA_DOMAIN, None, timeout=60)
        results = client.get(SOLAR_PRODUCTION_DATASET, limit=limit)
        client.close()
        df = pd.DataFrame.from_records(results)
    except Exception as exc:
        logger.error("Failed to fetch solar production data from Socrata API: %s", exc)
        if cache_path.exists():
            logger.warning("Falling back to cached solar production data.")
            return pd.read_csv(cache_path)
        logger.warning("No cached data available. Generating synthetic solar production data for demonstration...")
        df = _generate_synthetic_production()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Fetched and cached %d records to %s", len(df), cache_path)
    return df


def fetch_solar_sites(
    limit: int = 10000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch solar facility site information from Calgary Open Data.

    Parameters
    ----------
    limit : int
        Maximum number of records to retrieve.
    force_refresh : bool
        If True, bypass the local cache and re-download.

    Returns
    -------
    pd.DataFrame
        Raw solar site data.
    """
    cache_path = DATA_DIR / "solar_sites.csv"

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached solar sites data from %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Fetching solar sites data from Calgary Open Data...")
    try:
        client = Socrata(CALGARY_OPEN_DATA_DOMAIN, None, timeout=60)
        results = client.get(SOLAR_SITES_DATASET, limit=limit)
        client.close()
        df = pd.DataFrame.from_records(results)
    except Exception as exc:
        logger.error("Failed to fetch solar sites data from Socrata API: %s", exc)
        if cache_path.exists():
            logger.warning("Falling back to cached solar sites data.")
            return pd.read_csv(cache_path)
        logger.warning("No cached data available. Generating synthetic solar sites data for demonstration...")
        df = _generate_synthetic_sites()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Fetched and cached %d records to %s", len(df), cache_path)
    return df


def _generate_synthetic_production() -> pd.DataFrame:
    """
    Generate synthetic solar production data for demonstration purposes.

    Returns
    -------
    pd.DataFrame
        Synthetic production data matching the expected schema.
    """
    np.random.seed(42)

    facilities = [
        ("Southland Leisure Centre", "1", "2000 Southland Dr SW"),
        ("Bearspaw Water Treatment", "2", "16015 Bearspaw Dam Rd NW"),
        ("Bonnybrook Wastewater", "3", "5850 72 Ave SE"),
        ("Manchester Building", "4", "510 77 Ave SE"),
        ("Whitehorn Multi-Services", "5", "3500 32 Ave NE"),
        ("Village Square Leisure Centre", "6", "2623 56 St NE"),
        ("Fire Station 17", "7", "7 Mahogany Plaza SE"),
        ("Eau Claire Market", "8", "200 Barclay Parade SW"),
        ("Rocky Ridge Recreation", "9", "11300 Rocky Ridge Rd NW"),
        ("Quarry Park Recreation", "10", "108 Quarry Park Rd SE"),
    ]

    # Monthly solar irradiance pattern for Calgary (relative scale)
    monthly_solar_factor = {
        1: 0.25, 2: 0.35, 3: 0.55, 4: 0.70, 5: 0.85, 6: 0.95,
        7: 1.00, 8: 0.90, 9: 0.70, 10: 0.50, 11: 0.30, 12: 0.20,
    }

    records = []
    for name, fid, address in facilities:
        base_capacity = np.random.uniform(5000, 50000)
        for year in range(2018, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 10:
                    continue
                solar_factor = monthly_solar_factor[month]
                noise = np.random.normal(1.0, 0.08)
                year_trend = 1 + 0.02 * (year - 2018)
                production = base_capacity * solar_factor * noise * year_trend
                production = max(0, production)
                records.append({
                    "facility_name": name,
                    "id": fid,
                    "facility_address": address,
                    "period": f"{year}-{month:02d}",
                    "solar_pv_production_kwh": round(production, 2),
                })

    return pd.DataFrame(records)


def _generate_synthetic_sites() -> pd.DataFrame:
    """
    Generate synthetic solar facility site data for demonstration.

    Returns
    -------
    pd.DataFrame
        Synthetic site data.
    """
    facilities = [
        ("Southland Leisure Centre", "1", "2000 Southland Dr SW", 51.0010, -114.0718, 150),
        ("Bearspaw Water Treatment", "2", "16015 Bearspaw Dam Rd NW", 51.1120, -114.2015, 200),
        ("Bonnybrook Wastewater", "3", "5850 72 Ave SE", 50.9985, -114.0200, 300),
        ("Manchester Building", "4", "510 77 Ave SE", 50.9950, -114.0350, 100),
        ("Whitehorn Multi-Services", "5", "3500 32 Ave NE", 51.0780, -113.9850, 120),
        ("Village Square Leisure Centre", "6", "2623 56 St NE", 51.0730, -113.9600, 180),
        ("Fire Station 17", "7", "7 Mahogany Plaza SE", 50.9300, -113.9700, 80),
        ("Eau Claire Market", "8", "200 Barclay Parade SW", 51.0540, -114.0700, 90),
        ("Rocky Ridge Recreation", "9", "11300 Rocky Ridge Rd NW", 51.1250, -114.2300, 160),
        ("Quarry Park Recreation", "10", "108 Quarry Park Rd SE", 50.9800, -114.0100, 140),
    ]

    records = []
    for name, fid, address, lat, lon, capacity in facilities:
        records.append({
            "facility_name": name,
            "id": fid,
            "facility_address": address,
            "latitude": lat,
            "longitude": lon,
            "installed_capacity_kw": capacity,
            "installation_year": np.random.choice([2016, 2017, 2018, 2019]),
        })

    return pd.DataFrame(records)


def preprocess_production(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw solar production data.

    Parses the period column to datetime, extracts temporal features,
    converts production to numeric, and creates cyclical month encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Raw solar production data.

    Returns
    -------
    pd.DataFrame
        Preprocessed production data with added features.
    """
    df = df.copy()

    # Parse period to datetime (YYYY-MM format)
    df["period_dt"] = pd.to_datetime(df["period"], format="%Y-%m", errors="coerce")

    # Extract year and month
    df["year"] = df["period_dt"].dt.year
    df["month"] = df["period_dt"].dt.month

    # Convert production to numeric
    df["solar_pv_production_kwh"] = pd.to_numeric(
        df["solar_pv_production_kwh"], errors="coerce"
    )

    # Drop rows with missing critical values
    df = df.dropna(subset=["period_dt", "solar_pv_production_kwh"])

    # Cyclical encoding of month (sin/cos)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Sort by facility and date for proper rolling/lag calculations
    df = df.sort_values(["facility_name", "period_dt"]).reset_index(drop=True)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling average production per facility.

    Computes 3-month, 6-month, and 12-month rolling averages
    of solar PV production for each facility.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed production data sorted by facility and date.

    Returns
    -------
    pd.DataFrame
        Data with rolling average columns added.
    """
    df = df.copy()

    for window in [3, 6, 12]:
        col_name = f"rolling_avg_{window}m"
        df[col_name] = (
            df.groupby("facility_name")["solar_pv_production_kwh"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for solar production per facility.

    Adds 1-month, 3-month, and 12-month lag values of
    solar PV production for each facility.

    Parameters
    ----------
    df : pd.DataFrame
        Production data sorted by facility and date.

    Returns
    -------
    pd.DataFrame
        Data with lag feature columns added.
    """
    df = df.copy()

    for lag in [1, 3, 12]:
        col_name = f"lag_{lag}m"
        df[col_name] = (
            df.groupby("facility_name")["solar_pv_production_kwh"]
            .transform(lambda x: x.shift(lag))
        )

    return df


def compute_facility_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate statistics per facility.

    Computes mean, median, standard deviation, min, max, and
    total production for each facility.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed production data.

    Returns
    -------
    pd.DataFrame
        Aggregated statistics per facility.
    """
    stats = (
        df.groupby("facility_name")["solar_pv_production_kwh"]
        .agg(["mean", "median", "std", "min", "max", "sum", "count"])
        .rename(columns={
            "mean": "avg_monthly_kwh",
            "median": "median_monthly_kwh",
            "std": "std_monthly_kwh",
            "min": "min_monthly_kwh",
            "max": "max_monthly_kwh",
            "sum": "total_kwh",
            "count": "num_months",
        })
        .reset_index()
    )
    return stats


def load_and_prepare_data(force_refresh: bool = False) -> dict:
    """
    Full data loading and preparation pipeline.

    Fetches data, preprocesses, engineers features, and computes
    facility-level statistics.

    Parameters
    ----------
    force_refresh : bool
        If True, re-download data from the API.

    Returns
    -------
    dict
        Dictionary containing:
        - 'production': full featured production DataFrame
        - 'sites': solar site information DataFrame
        - 'facility_stats': per-facility aggregated statistics
    """
    # Fetch raw data
    production_raw = fetch_solar_production(force_refresh=force_refresh)
    sites = fetch_solar_sites(force_refresh=force_refresh)

    # Preprocess
    production = preprocess_production(production_raw)

    # Feature engineering
    production = add_rolling_features(production)
    production = add_lag_features(production)

    # Facility statistics
    facility_stats = compute_facility_stats(production)

    return {
        "production": production,
        "sites": sites,
        "facility_stats": facility_stats,
    }


if __name__ == "__main__":
    data = load_and_prepare_data()
    print(f"Production records: {len(data['production'])}")
    print(f"Facilities: {data['production']['facility_name'].nunique()}")
    print(f"\nFacility stats:\n{data['facility_stats']}")
