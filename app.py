"""
Calgary Solar Energy Production Forecaster — Streamlit Dashboard.

Interactive application for exploring Calgary's municipal solar PV
production data, analyzing facility-level performance, generating
multi-step forecasts, and reviewing model performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_prepare_data, compute_facility_stats
from src.model import (
    train_and_evaluate,
    generate_forecast,
    get_models,
    FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Calgary Solar Energy Production Forecaster",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading solar production data...")
def load_data():
    """Load and cache all project data."""
    return load_and_prepare_data()


@st.cache_resource(show_spinner="Training forecasting models...")
def train_models(production_json: str):
    """Train models and cache the results."""
    production = pd.read_json(production_json)
    production["period_dt"] = pd.to_datetime(production["period_dt"])
    results = train_and_evaluate(production, test_months=12)
    return results


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Solar Dashboard",
        "Facility Analysis",
        "Production Forecast",
        "Model Performance",
        "About",
    ],
)

# Load data
data = load_data()
production = data["production"]
sites = data["sites"]
facility_stats = data["facility_stats"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def format_kwh(value: float) -> str:
    """Format kWh values with thousands separators."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:,.2f} MWh"
    return f"{value:,.0f} kWh"


# ===========================================================================
# PAGE: Solar Dashboard
# ===========================================================================
if page == "Solar Dashboard":
    st.title("Calgary Solar Energy Production Dashboard")
    st.markdown(
        "Overview of solar PV production across City of Calgary municipal facilities."
    )

    # --- Key metrics ---
    col1, col2, col3, col4 = st.columns(4)

    total_facilities = production["facility_name"].nunique()
    total_production = production["solar_pv_production_kwh"].sum()
    avg_monthly = production.groupby("period_dt")["solar_pv_production_kwh"].sum().mean()
    latest_month_prod = (
        production[production["period_dt"] == production["period_dt"].max()]
        ["solar_pv_production_kwh"].sum()
    )

    col1.metric("Total Facilities", total_facilities)
    col2.metric("Total Production", format_kwh(total_production))
    col3.metric("Avg Monthly (All)", format_kwh(avg_monthly))
    col4.metric("Latest Month", format_kwh(latest_month_prod))

    st.markdown("---")

    # --- Citywide production over time ---
    st.subheader("Citywide Solar Production Over Time")

    citywide = (
        production.groupby("period_dt")["solar_pv_production_kwh"]
        .sum()
        .reset_index()
        .sort_values("period_dt")
    )

    fig_city = px.line(
        citywide,
        x="period_dt",
        y="solar_pv_production_kwh",
        labels={
            "period_dt": "Month",
            "solar_pv_production_kwh": "Production (kWh)",
        },
        title="Total Monthly Solar PV Production — All Facilities",
    )
    fig_city.update_layout(hovermode="x unified")
    st.plotly_chart(fig_city, use_container_width=True)

    # --- Production by facility (stacked area) ---
    st.subheader("Production by Facility")

    facility_monthly = (
        production.groupby(["period_dt", "facility_name"])["solar_pv_production_kwh"]
        .sum()
        .reset_index()
        .sort_values("period_dt")
    )

    fig_stacked = px.area(
        facility_monthly,
        x="period_dt",
        y="solar_pv_production_kwh",
        color="facility_name",
        labels={
            "period_dt": "Month",
            "solar_pv_production_kwh": "Production (kWh)",
            "facility_name": "Facility",
        },
        title="Monthly Production by Facility (Stacked Area)",
    )
    fig_stacked.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_stacked, use_container_width=True)

    # --- Facility ranking ---
    st.subheader("Facility Production Ranking")

    fig_rank = px.bar(
        facility_stats.sort_values("total_kwh", ascending=True),
        x="total_kwh",
        y="facility_name",
        orientation="h",
        labels={
            "total_kwh": "Total Production (kWh)",
            "facility_name": "Facility",
        },
        title="Cumulative Production by Facility",
    )
    st.plotly_chart(fig_rank, use_container_width=True)


# ===========================================================================
# PAGE: Facility Analysis
# ===========================================================================
elif page == "Facility Analysis":
    st.title("Facility Analysis")
    st.markdown("Detailed analysis of solar production for individual facilities.")

    facilities_list = sorted(production["facility_name"].unique())
    selected_facility = st.selectbox("Select Facility", facilities_list)

    fac_data = (
        production[production["facility_name"] == selected_facility]
        .sort_values("period_dt")
        .copy()
    )

    # --- Facility stats card ---
    st.subheader(f"Statistics: {selected_facility}")
    fac_stats = facility_stats[facility_stats["facility_name"] == selected_facility]

    if not fac_stats.empty:
        row = fac_stats.iloc[0]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Production", format_kwh(row["total_kwh"]))
        mc2.metric("Avg Monthly", format_kwh(row["avg_monthly_kwh"]))
        mc3.metric("Peak Month", format_kwh(row["max_monthly_kwh"]))
        mc4.metric("Months Recorded", int(row["num_months"]))

    st.markdown("---")

    # --- Monthly production trend ---
    st.subheader("Monthly Production Trend")

    fig_trend = px.line(
        fac_data,
        x="period_dt",
        y="solar_pv_production_kwh",
        labels={
            "period_dt": "Month",
            "solar_pv_production_kwh": "Production (kWh)",
        },
        title=f"Monthly Solar Production — {selected_facility}",
    )
    fig_trend.update_layout(hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Seasonal pattern ---
    st.subheader("Seasonal Pattern (Average by Month)")

    seasonal = (
        fac_data.groupby("month")["solar_pv_production_kwh"]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    seasonal["month_name"] = seasonal["month"].apply(lambda m: month_names[m - 1])

    fig_season = px.bar(
        seasonal,
        x="month_name",
        y="solar_pv_production_kwh",
        labels={
            "month_name": "Month",
            "solar_pv_production_kwh": "Avg Production (kWh)",
        },
        title=f"Average Monthly Production — {selected_facility}",
        color="solar_pv_production_kwh",
        color_continuous_scale="YlOrRd",
    )
    fig_season.update_layout(xaxis=dict(categoryorder="array", categoryarray=month_names))
    st.plotly_chart(fig_season, use_container_width=True)

    # --- Year-over-year comparison ---
    st.subheader("Year-over-Year Comparison")

    fac_data["year_str"] = fac_data["year"].astype(str)

    fig_yoy = px.line(
        fac_data,
        x="month",
        y="solar_pv_production_kwh",
        color="year_str",
        labels={
            "month": "Month",
            "solar_pv_production_kwh": "Production (kWh)",
            "year_str": "Year",
        },
        title=f"Year-over-Year Production — {selected_facility}",
    )
    fig_yoy.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_yoy, use_container_width=True)


# ===========================================================================
# PAGE: Production Forecast
# ===========================================================================
elif page == "Production Forecast":
    st.title("Solar Production Forecast")
    st.markdown(
        "Generate multi-month production forecasts for a selected facility."
    )

    # Sidebar controls
    facilities_list = sorted(production["facility_name"].unique())
    selected_facility = st.selectbox("Select Facility", facilities_list)
    horizon = st.selectbox("Forecast Horizon (months)", [6, 12, 24], index=1)

    if st.button("Generate Forecast"):
        with st.spinner("Training model and generating forecast..."):
            # Train models
            eval_results = train_models(production.to_json())
            best_model = eval_results["best_model"]
            best_name = eval_results["best_model_name"]

            st.success(f"Using best model: **{best_name}**")

            # Generate forecast
            forecast_df = generate_forecast(
                best_model, production, selected_facility, n_months=horizon
            )

        # Prepare historical data for the facility
        fac_hist = (
            production[production["facility_name"] == selected_facility]
            .sort_values("period_dt")
            .copy()
        )

        # Combined plot: historical + forecast
        st.subheader(f"Forecast: {selected_facility} ({horizon} months)")

        fig_fc = go.Figure()

        fig_fc.add_trace(go.Scatter(
            x=fac_hist["period_dt"],
            y=fac_hist["solar_pv_production_kwh"],
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4"),
        ))

        fig_fc.add_trace(go.Scatter(
            x=forecast_df["period_dt"],
            y=forecast_df["predicted_kwh"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
            marker=dict(size=6),
        ))

        fig_fc.update_layout(
            title=f"Solar Production Forecast — {selected_facility}",
            xaxis_title="Month",
            yaxis_title="Production (kWh)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Overlay seasonal pattern on forecast
        st.subheader("Forecast with Seasonal Context")

        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        seasonal_avg = (
            fac_hist.groupby("month")["solar_pv_production_kwh"]
            .mean()
            .to_dict()
        )

        forecast_df["seasonal_avg"] = forecast_df["month"].map(seasonal_avg)
        forecast_df["month_name"] = forecast_df["month"].apply(
            lambda m: month_names[m - 1]
        )

        fig_ctx = go.Figure()

        fig_ctx.add_trace(go.Bar(
            x=forecast_df["period_dt"],
            y=forecast_df["seasonal_avg"],
            name="Historical Seasonal Avg",
            marker_color="rgba(31,119,180,0.3)",
        ))

        fig_ctx.add_trace(go.Scatter(
            x=forecast_df["period_dt"],
            y=forecast_df["predicted_kwh"],
            mode="lines+markers",
            name="Forecasted",
            line=dict(color="#ff7f0e", width=2),
        ))

        fig_ctx.update_layout(
            title="Forecast vs. Historical Seasonal Average",
            xaxis_title="Month",
            yaxis_title="Production (kWh)",
            hovermode="x unified",
            barmode="overlay",
        )
        st.plotly_chart(fig_ctx, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Data")
        display_df = forecast_df[["period_dt", "month_name", "predicted_kwh"]].copy()
        display_df.columns = ["Date", "Month", "Predicted kWh"]
        display_df["Predicted kWh"] = display_df["Predicted kWh"].apply(
            lambda v: f"{v:,.0f}"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select a facility and forecast horizon, then click **Generate Forecast**.")


# ===========================================================================
# PAGE: Model Performance
# ===========================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Comparison of forecasting models and residual analysis.")

    with st.spinner("Training and evaluating models..."):
        eval_results = train_models(production.to_json())

    results = eval_results["results"]
    best_name = eval_results["best_model_name"]
    test_data = eval_results["test_data"]

    # --- Model comparison table ---
    st.subheader("Model Comparison")

    comparison_rows = []
    for r in results:
        m = r["metrics"]
        comparison_rows.append({
            "Model": r["model_name"],
            "MAE (kWh)": f"{m['MAE']:,.1f}",
            "RMSE (kWh)": f"{m['RMSE']:,.1f}",
            "R-squared": f"{m['R2']:.4f}",
            "MAPE (%)": f"{m['MAPE']:.2f}",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.info(f"Best model by RMSE: **{best_name}**")

    st.markdown("---")

    # --- Actual vs Predicted scatter ---
    st.subheader("Actual vs. Predicted (Best Model)")

    fig_scatter = px.scatter(
        test_data,
        x="solar_pv_production_kwh",
        y="predicted_kwh",
        color="facility_name",
        labels={
            "solar_pv_production_kwh": "Actual (kWh)",
            "predicted_kwh": "Predicted (kWh)",
            "facility_name": "Facility",
        },
        title=f"Actual vs. Predicted — {best_name}",
        opacity=0.7,
    )

    # Add perfect prediction line
    max_val = max(
        test_data["solar_pv_production_kwh"].max(),
        test_data["predicted_kwh"].max(),
    )
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="gray", dash="dash"),
    ))

    fig_scatter.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Residual analysis ---
    st.subheader("Residual Analysis")

    test_data["residual"] = (
        test_data["solar_pv_production_kwh"] - test_data["predicted_kwh"]
    )

    fig_resid_hist = px.histogram(
        test_data,
        x="residual",
        nbins=40,
        labels={"residual": "Residual (kWh)"},
        title="Distribution of Residuals",
        color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(fig_resid_hist, use_container_width=True)

    fig_resid_time = px.scatter(
        test_data,
        x="period_dt",
        y="residual",
        color="facility_name",
        labels={
            "period_dt": "Month",
            "residual": "Residual (kWh)",
            "facility_name": "Facility",
        },
        title="Residuals Over Time",
        opacity=0.7,
    )
    fig_resid_time.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_resid_time.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_resid_time, use_container_width=True)


# ===========================================================================
# PAGE: About
# ===========================================================================
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Calgary's Solar Energy Goals

    The City of Calgary has been expanding its portfolio of solar photovoltaic (PV)
    installations across municipal buildings and facilities as part of its broader
    climate action strategy. Accurate forecasting of solar energy production helps
    the city:

    - **Optimize energy purchasing** by anticipating how much solar energy will
      offset grid electricity consumption each month.
    - **Plan maintenance schedules** by identifying facilities that underperform
      relative to expectations.
    - **Track progress** toward renewable energy targets and report to stakeholders.
    - **Budget effectively** by projecting future energy cost savings from solar
      installations.

    ## Methodology

    This forecaster uses supervised machine learning to predict monthly solar PV
    production (kWh) for each municipal facility. The approach includes:

    1. **Cyclical time encoding**: Month is encoded as sine and cosine components
       to capture the circular nature of seasonal patterns.
    2. **Autoregressive features**: Lag values (1, 3, and 12 months) allow the
       model to learn from recent and year-ago production levels.
    3. **Rolling statistics**: Moving averages (3, 6, 12 months) smooth out noise
       and capture medium-term trends.
    4. **Temporal train/test split**: The most recent 12 months are held out for
       evaluation to prevent data leakage from the future.
    5. **Iterative forecasting**: For multi-step horizons, each month's prediction
       is fed back as input for subsequent months.

    Three models are compared:
    - **Ridge Regression** — L2-regularized linear model (baseline)
    - **Random Forest** — bagged ensemble of decision trees
    - **XGBoost** — gradient-boosted decision trees

    ## Data Source

    All data is sourced from the **City of Calgary Open Data Portal**:

    | Dataset | Identifier |
    |---|---|
    | Solar PV Production | `iric-4rrc` |
    | Solar Facility Sites | `tbsv-89ps` |

    Data is accessed via the Socrata Open Data API (SODA) using the `sodapy`
    Python client. A local CSV cache is maintained to minimize API calls.

    ## Technology Stack

    - **Python** — core language
    - **Streamlit** — interactive web dashboard
    - **Plotly** — interactive visualizations
    - **scikit-learn / XGBoost** — machine learning models
    - **pandas / NumPy** — data manipulation
    - **sodapy** — Calgary Open Data API client
    """)

    st.markdown("---")
    st.markdown(
        "*Built as part of the Calgary Data Portfolio project. "
        "Data provided by the City of Calgary Open Data Portal.*"
    )
