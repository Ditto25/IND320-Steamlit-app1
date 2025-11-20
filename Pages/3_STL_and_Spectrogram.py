import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import toml
import numpy as np
from functools import lru_cache

# Optional imports for STL and Spectrogram
try:
    # Import dynamically to avoid static analysis errors when statsmodels is not installed
    import importlib
    stats_seasonal = importlib.import_module("statsmodels.tsa.seasonal")
    STL = getattr(stats_seasonal, "STL")
    _STL_AVAILABLE = True
except Exception:
    STL = None
    _STL_AVAILABLE = False

try:
    from scipy import signal
    _SCIPY_AVAILABLE = True
except Exception:
    signal = None
    _SCIPY_AVAILABLE = False

# --- Page Config ---
st.set_page_config(page_title="Advanced Time Series Analysis", layout="wide")

# --- MongoDB Connection ---
@lru_cache(maxsize=1)
def init_connection():
    secrets = toml.load(".streamlit/secrets.toml")
    uri = secrets["MONGO"]["uri"]
    return MongoClient(uri, server_api=ServerApi('1'))

@lru_cache(maxsize=1)
def load_production_data():
    client = init_connection()
    db = client['Database']
    collection = db['data']
    records = list(collection.find({}, {'_id': 0}))
    if not records:
        raise ValueError("No data found in MongoDB! Please insert data first.")
    df = pd.DataFrame(records)
    df['startTime_parsed'] = pd.to_datetime(df['startTime'], utc=True)
    df['endTime_parsed'] = pd.to_datetime(df['endTime'], utc=True)
    return df

# --- STL Decomposition ---
def stl_analysis(df, price_area, production_group, period=24, seasonal=7, trend=None, robust=False):
    if not _STL_AVAILABLE:
        return None, "⚠️ STL not available (install statsmodels)"
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    if filtered.empty:
        return None, "No data available for this combination"
    ts = pd.Series(filtered['quantityKwh'].values, index=filtered['startTime_parsed']).ffill().bfill()
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    axes[0].plot(ts, color='black', linewidth=1)
    axes[0].set_title(f"{production_group} — {price_area}", fontsize=14, fontweight='bold')
    labels = ['Trend', 'Seasonal', 'Residual']
    colors = ['blue', 'green', 'red']
    for i, comp in enumerate([result.trend, result.seasonal, result.resid]):
        axes[i+1].plot(ts.index, comp, color=colors[i])
        axes[i+1].set_ylabel(labels[i], fontsize=11, fontweight='bold')
        axes[i+1].grid(True, alpha=0.3)
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig, None

# --- Spectrogram ---
def spectrogram_analysis(df, price_area, production_group, window_length=168, window_overlap=84):
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    if filtered.empty:
        return None, "No data available for this combination"
    production = pd.Series(filtered['quantityKwh'].values).ffill().bfill().values
    if _SCIPY_AVAILABLE:
        f, t, Sxx = signal.spectrogram(production, fs=1.0, window='hann',
                                       nperseg=window_length, noverlap=window_overlap)
    else:
        return None, "⚠️ SciPy not available (install scipy)"
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (cycles/hour)', fontsize=12, fontweight='bold')
    ax.set_title(f"{production_group} — {price_area}", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig, None

# --- Page Layout ---
st.title("📊 Advanced Time Series Analysis")
st.caption("Analyze electricity production patterns with STL decomposition and spectrograms.")
st.markdown("---")

try:
    df = load_production_data()
    price_areas = sorted(df['priceArea'].unique())
    production_groups = sorted(df['productionGroup'].unique())

    tab1, tab2 = st.tabs(["🧩 STL Decomposition", "🎵 Spectrogram"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        stl_area = col1.selectbox("Price Area", price_areas, key="stl_area")
        stl_group = col2.selectbox("Production Group", production_groups, key="stl_group")
        stl_period = col3.number_input("Seasonal Period", 2, 720, 24)

        col4, col5 = st.columns(2)
        stl_seasonal = col4.slider("Seasonal Smoothing", 3, 25, 7, step=2)
        stl_robust = col5.checkbox("Robust Fitting", True)

        if st.button("Run STL Analysis", key="stl_button"):
            with st.spinner("Running STL Decomposition..."):
                fig, error = stl_analysis(df, stl_area, stl_group, stl_period, stl_seasonal, robust=stl_robust)
                if error:
                    st.error(error)
                else:
                    st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)
        spec_area = col1.selectbox("Price Area", price_areas, key="spec_area")
        spec_group = col2.selectbox("Production Group", production_groups, key="spec_group")

        col3, col4 = st.columns(2)
        spec_window = col3.slider("Window Length (hours)", 24, 720, 168, step=24)
        spec_overlap = col4.slider("Window Overlap (hours)", 0, int(spec_window * 0.9), int(spec_window * 0.5), step=12)

        if st.button("Create Spectrogram", key="spec_button"):
            with st.spinner("Computing Spectrogram..."):
                fig, error = spectrogram_analysis(df, spec_area, spec_group, spec_window, spec_overlap)
                if error:
                    st.error(error)
                else:
                    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)
