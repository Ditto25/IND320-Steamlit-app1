import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go 
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit.components.v1 as components 
import io 
from reportlab.lib.pagesizes import letter 
from reportlab.pdfgen import canvas 
import plotly.io as pio 
import requests 

# Adjusted import path to match previous files (src -> utils) if necessary
from utils.Data_loader import ( 
    load_elhub_production_data,
    load_elhub_consumption_data,
)

# ---------- Helper functions ---------- #

st.set_page_config(page_title="Energy Forecasting", layout="wide")

def get_target_series(df: pd.DataFrame, area: str, group: str) -> pd.Series:
    """
    Filter Elhub data for one price area and one group, return a clean hourly series.
    
    NOTE: Uses the CamelCase column names correct for the database.
    """
    sub = df[(df["priceArea"] == area) & (df["group"] == group)].copy()

    if sub.empty:
        return pd.Series(dtype="float64")

    sub["startTime"] = pd.to_datetime(sub["startTime"])
    sub = sub.sort_values("startTime")

    series = (
        sub.set_index("startTime")["quantityKwh"] 
        .astype("float64")
        .resample("h") 
        .sum(min_count=1)
    )
    series = series.interpolate(limit_direction="both")
    return series

# --- NY HELPER-FUNKSJON ---
def get_model_metrics(results, ylabel: str) -> dict:
    """
    Calculates key diagnostic metrics (AIC, BIC, RMSE, etc.) from SARIMAXResults object.
    
    Args:
        results: The fitted SARIMAXResults object.
        ylabel: The string used for the Y-axis label (to extract the unit).
        
    Returns:
        A dictionary containing the calculated metrics.
    """
    # 1. Error calculation (RMSE)
    residuals = results.resid
    rmse = float(np.sqrt(np.mean(residuals**2)))
    
    # 2. Extract unit
    unit_match = ylabel.split('(')
    unit = unit_match[1].replace(')','') if len(unit_match) > 1 else "units"
    
    # 3. Compile metrics
    metrics = {
        "aic": results.aic,
        "bic": results.bic,
        "hqic": results.hqic,
        "llf": results.llf, # Log Likelihood
        "rmse": rmse,
        "unit": unit,
    }
    return metrics
# -------------------------

def build_forecast_figure(
    train_series: pd.Series,
    fitted: pd.Series | None,
    forecast: pd.Series,
    conf_int: pd.DataFrame,
    title: str,
    ylabel: str,
) -> go.Figure:
    """
    Create a Plotly figure with training data, fitted values, forecast and confidence interval.
    """
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    fig = go.Figure()

    # Training data
    fig.add_trace(
        go.Scatter(
            x=train_series.index,
            y=train_series.values,
            mode="lines",
            name="Training data",
            line=dict(color="blue"),
        )
    )

    # In-sample fitted values (if available)
    if fitted is not None and not fitted.empty:
        fig.add_trace(
            go.Scatter(
                x=fitted.index,
                y=fitted.values,
                mode="lines",
                name="Fitted (in-sample)",
                line=dict(width=1, dash="dot", color="green"),
            )
        )

    # Confidence interval band
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=upper.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=lower.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Confidence interval",
            fillcolor="rgba(255,165,0,0.2)",
            hoverinfo="skip",
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines",
            name="Forecast",
            line=dict(color="orange"),
        )
    )

    # Vertical line at training end
    fig.add_vline(
        x=train_series.index[-1],
        line_width=1,
        line_dash="dot",
        line_color="gray",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig

# --- SIMULATED WEATHER DATA FETCH (Unchanged) ---
@st.cache_data
def fetch_simulated_weather(start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str):
    """Simulates daily or hourly weather data for a given period."""
    if freq == "h":
        idx = pd.date_range(start=start_date.floor('h'), end=end_date.ceil('h'), freq='h')
        scale = 1
    else:
        idx = pd.date_range(start=start_date.floor('D'), end=end_date.ceil('D'), freq='D')
        scale = 24
        
    np.random.seed(42) # Ensure reproducible simulation
    df = pd.DataFrame(index=idx)
    
    # Simulate data
    df["Temperature (°C)"] = 10 + 5 * np.sin(np.linspace(0, 2 * np.pi, len(idx)) * 7 / scale) + np.random.randn(len(idx)) * 2
    df["Precipitation (mm)"] = np.random.rand(len(idx)) * 5
    df["Wind Speed (m/s)"] = 5 + np.random.rand(len(idx)) * 5
    
    df.index.name = "time"
    return df.reset_index()


# ---------- Streamlit page (Main flow adjusted to use the helper) ---------- #

st.title("Energy Forecasting with Exogenous Variables")
st.markdown(
    """
This page trains a **SARIMAX** model, optionally incorporating **exogenous variables** (like weather data) for improved forecasts.

- Choose dataset, price area, group, training period, and forecast horizon.
- Select weather variables to include in the model.
"""
)

# Load data once (cached in data_loader)
prod_df = load_elhub_production_data()
cons_df = load_elhub_consumption_data()

# ---------- Controls (Unchanged) ---------- #

st.header("Configuration")

# Dataset selection
dataset_type = st.radio(
    "Dataset",
    ["Production", "Consumption"],
    horizontal=True,
)

if dataset_type == "Production":
    df = prod_df.copy()
    AVAILABLE_YEARS = [2022, 2023, 2024]
else:
    df = cons_df.copy()
    AVAILABLE_YEARS = [2021, 2022, 2023, 2024]

# Price areas
areas = sorted(df["priceArea"].dropna().unique().tolist()) 
if not areas:
    st.error("No price areas available in the selected dataset.")
    st.stop()

# Set initial session state for area if missing
if "priceArea" not in st.session_state:
    st.session_state["priceArea"] = areas[0]

default_index = (
    areas.index(st.session_state["priceArea"])
    if st.session_state["priceArea"] in areas
    else 0
)

col_area, col_group = st.columns(2)
with col_area:
    area = st.selectbox(
        "Price area",
        areas,
        index=default_index,
    )
    st.session_state["priceArea"] = area

# Groups
groups = sorted(df["group"].dropna().unique().tolist()) 
with col_group:
    group = st.selectbox("Group", groups)

target_hourly = get_target_series(df, area=area, group=group)

if target_hourly.empty:
    st.header("Forecast")
    st.warning("No data found for this combination of dataset, price area and group.")
    st.stop()
    
st.subheader("Frequency and Training Period")

# Frequency selection
freq_label = st.selectbox("Frequency", ["Hourly", "Daily"], index=0)
freq = "h" if freq_label == "Hourly" else "D"
ylabel = f"Quantity (kWh{'/day' if freq=='D' else ''})"


min_date_actual = target_hourly.index.min().date()
max_date_actual = target_hourly.index.max().date()

# Default date calculation (using the latest available year)
default_year = AVAILABLE_YEARS[-1]
start_of_year = pd.to_datetime(f'{default_year}-01-01').date()
end_of_year = pd.to_datetime(f'{default_year}-12-31').date()

start_date_default = max(min_date_actual, start_of_year)
end_date_default = min(max_date_actual, end_of_year)
if start_date_default > max_date_actual:
    start_date_default = min_date_actual
    end_date_default = max_date_actual

col_start, col_end = st.columns(2)
with col_start:
    start_date = st.date_input(
        "Start date",
        value=start_date_default, 
        min_value=min_date_actual,
        max_value=max_date_actual,
    )
with col_end:
    end_date = st.date_input(
        "End date",
        value=end_date_default, 
        min_value=min_date_actual,
        max_value=max_date_actual,
    )
    
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# --- Prepare Energy Time Series ---
train_series_full = target_hourly.loc[str(start_date) : str(end_date)]

if train_series_full.empty:
    st.error("No data in the selected training period.")
    st.stop()

# Resample/Aggregate
if freq == "D":
    train_series = train_series_full.resample("D").sum().dropna()
else:
    train_series = train_series_full.asfreq("h").dropna()


# --- Exogenous Variable Selection (Unchanged) ---
st.subheader("Exogenous Variables (Weather)")

# Simulate fetching weather data for the required period
start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)
df_meteo = fetch_simulated_weather(start_ts, end_ts + pd.Timedelta(days=365), freq) 

meteo_columns = [c for c in df_meteo.columns if c != "time"]

selected_exog = st.multiselect(
    "Select weather variables for SARIMAX",
    meteo_columns,
    default=["Temperature (°C)"]
)

# --- SARIMAX parameters (Unchanged) ---
st.subheader("SARIMAX Parameters")

with st.expander("Show SARIMAX settings"):
    if freq == "h":
        default_p, default_d, default_q = 1, 0, 1
        default_P, default_D, default_Q, default_s = 0, 1, 1, 24   
    else:  
        default_p, default_d, default_q = 1, 0, 1
        default_P, default_D, default_Q, default_s = 0, 1, 1, 7    

    col_p, col_d, col_q = st.columns(3)
    p = col_p.number_input("p (AR)", value=default_p, min_value=0, max_value=5, help="Autoregressive order")
    d = col_d.number_input("d (Diff)", value=default_d, min_value=0, max_value=2, help="Integrated order (Differencing)")
    q = col_q.number_input("q (MA)", value=default_q, min_value=0, max_value=5, help="Moving Average order")

    col_P, col_D, col_Q, col_s = st.columns(4)
    P = col_P.number_input("P (Seasonal AR)", value=default_P, min_value=0, max_value=5, help="Seasonal Autoregressive order")
    D = col_D.number_input("D (Seasonal Diff)", value=default_D, min_value=0, max_value=2, help="Seasonal Integrated order")
    Q = col_Q.number_input("Q (Seasonal MA)", value=default_Q, min_value=0, max_value=5, help="Seasonal Moving Average order")
    s = col_s.number_input("Seasonal period (s)", value=default_s, min_value=1, max_value=365, help="Number of time steps per seasonal cycle (e.g., 24 for hourly, 7 for daily)")


# Forecast horizon
st.subheader("Forecast Horizon")
default_h = 7 if freq == "D" else 24*7
max_h = 365 if freq == "D" else 24*60
horizon = st.number_input(
    f"Steps ahead ({freq_label})",
    min_value=1,
    max_value=max_h,
    value=default_h,
    step=1,
)

    
run_forecast = st.button("Train & Forecast")

# ---------- Forecast Execution (Integrated Logic - uses helper) ---------- #
st.header("Model Output")

if run_forecast:
    if train_series.empty or len(train_series) < 30:
        st.error("Training data is too short or empty. Please select a longer period.")
        st.stop()

    try:
        with st.spinner("Preparing data and fitting SARIMAX model..."):
            
            # --- Prepare Exogenous Data for Training ---
            if selected_exog:
                df_m_train = df_meteo.set_index("time")[selected_exog]
                exog_train = df_m_train.resample(freq).mean().reindex(train_series.index)
                exog_train = exog_train.interpolate(limit_direction="both")
                
                valid_index = train_series.index
                exog_train = exog_train.loc[valid_index].dropna(how='all')
                train_series = train_series.loc[valid_index] 
                
                if exog_train.empty:
                    st.warning("Exogenous data is empty after processing. Running model without it.")
                    exog_train = None
            else:
                exog_train = None
            
            
            # --- Model Fitting ---
            model = SARIMAX(
                train_series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
                freq=freq, 
            )
            results = model.fit(disp=False, maxiter=100) 
            st.success("Model successfully trained.")
            
            # --- Forecast Index and Exogenous Data for Forecast ---
            future_start = train_series.index[-1] + pd.Timedelta(seconds=1) if freq=='h' else train_series.index[-1] + pd.Timedelta(days=1)
            future_index = pd.date_range(
                start=future_start,
                periods=horizon,
                freq=freq
            )
            
            if selected_exog:
                df_m_future = df_meteo.set_index("time")[selected_exog].resample(freq).mean()
                exog_future = df_m_future.reindex(future_index).fillna(method='ffill')
                if exog_future.empty or exog_future.isnull().values.any():
                    st.warning("Future exogenous data missing or incomplete. Imputing with FFILL/Mean.")
                    exog_future = exog_future.fillna(exog_train.mean())
            else:
                exog_future = None
                
            # --- Generate Forecast ---
            forecast_res = results.get_forecast(steps=horizon, exog=exog_future)
            forecast_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()
            
            # --- PLOTTING ---
            title = (
                f"{dataset_type} forecast – {area}, {group} "
                f"({train_series.index.min().date()} to {train_series.index.max().date()}, "
                f"horizon {horizon} {freq_label.lower()} steps)"
            )
            
            fig = build_forecast_figure(
                train_series=train_series,
                fitted=results.fittedvalues,
                forecast=forecast_mean,
                conf_int=conf_int,
                title=title,
                ylabel=ylabel,
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- MODEL SUMMARY AND METRICS (Uses the new helper) ---
            
            # 1. Use the helper function to get metrics
            metrics = get_model_metrics(results, ylabel)
            
            st.subheader("Model Diagnostics and Summary")
            
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.metric(label="AIC", value=f"{metrics['aic']:,.2f}")
                st.metric(label="Log Likelihood", value=f"{metrics['llf']:,.2f}")
            with col_2:
                st.metric(label="BIC", value=f"{metrics['bic']:,.2f}")
                st.metric(label="HQIC", value=f"{metrics['hqic']:,.2f}")
            with col_3:
                st.metric(label="Training RMSE", value=f"{metrics['rmse']:,.2f} {metrics['unit']}")
                st.markdown(f"**Model:** SARIMAX({p}, {d}, {q})x({P}, {D}, {Q}, {s})")

            st.markdown("---")
            st.markdown("### SARIMAX Output Summary")
            st.code(
                results.summary().as_text(),
                language='text'
            )
            
            st.markdown("---")

            # --- PDF & PNG DOWNLOAD SECTION (Unchanged) ---
            st.subheader("Download Results")
            
            # 1. Create PDF of SARIMAX Summary
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            textobject = c.beginText(40, 750)
            textobject.setFont("Helvetica", 10)

            summary_str = results.summary().as_text()
            y_position = 750
            line_height = 12
            max_lines = 60 
            
            for i, line in enumerate(summary_str.splitlines()):
                if i > 0 and i % max_lines == 0:
                    c.drawText(textobject)
                    c.showPage()
                    textobject = c.beginText(40, 750)
                    textobject.setFont("Helvetica", 10)
                    y_position = 750 
                
                textobject.textLine(line)
            
            c.drawText(textobject)
            c.save()
            pdf_buffer.seek(0)

            # 2. Save Plot as PNG
            png_buffer = io.BytesIO()
            pio.write_image(fig, file=png_buffer, format='png')
            png_buffer.seek(0)
            
            col_pdf, col_png = st.columns(2)

            with col_pdf:
                st.download_button(
                    label="📥 Download SARIMAX Summary (PDF)",
                    data=pdf_buffer,
                    file_name=f"sarimax_summary_{dataset_type}_{area}.pdf",
                    mime="application/pdf"
                )

            with col_png:
                st.download_button(
                    label="📥 Download Forecast Plot (PNG)",
                    data=png_buffer,
                    file_name=f"sarimax_forecast_{dataset_type}_{area}.png",
                    mime="image/png"
                )


    except Exception as e:
        st.error(f"SARIMAX model fitting or forecasting failed: {e}")
        st.exception(e)