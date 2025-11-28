import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import toml
import numpy as np
from functools import lru_cache
import plotly.graph_objs as go
import plotly.subplots as sp
from utils.Data_loader import render_sidebar_info # OBS: Må importeres eller mockes hvis den ikke finnes
from scipy import signal # Importeres her siden den brukes direkte

# Optional imports for STL
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


# --- Initial Configuration ---
render_sidebar_info() # KALL DEN HER
st.set_page_config(page_title="Advanced Time Series Analysis", layout="wide")
st.title("Avansert Analyse")


# --- Utility Functions ---

def check_data_requirements(require_weather=False, require_coordinates=False, require_energy=False):
    """
    Sjekker om nødvendige data er tilgjengelig i session state.
    Stopper kjøringen og viser en advarsel hvis data mangler.
    
    Returnerer: True hvis alle krav er møtt, False ellers.
    """
    missing = []
    
    # Sjekk etter koordinater (satt av kartet)
    if require_coordinates:
        if ('map_lat' not in st.session_state or 
            st.session_state.get('map_lat') is None or 
            st.session_state.get('map_lat') == 63.5): # Bruk initialverdien som sjekk
            missing.append("lokasjonsvalg (klikk på kartet)")
    
    # Sjekk etter værdata (satt av side 2)
    if require_weather:
        if 'weather_data' not in st.session_state or st.session_state.weather_data is None:
            missing.append("nedlasting av værdata")
    
    # Sjekk etter energidata (satt av en data-loader)
    if require_energy:
        # Hvis du bruker separate funksjoner for produksjon/forbruk, sjekk begge:
        if ('production_data' not in st.session_state and 
            'consumption_data' not in st.session_state):
            missing.append("energidata")
    
    if missing:
        st.warning(f"⚠️ Mangler: {', '.join(missing)}. Vennligst besøk **Kartvisualisering** eller **Weather Data Downloader** siden først.")
        return False
    return True

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

# --- STL Decomposition (Plotly) ---
def stl_analysis(df, price_area, production_group, period=24, seasonal=7, trend=None, robust=False):
    if not _STL_AVAILABLE:
        return None, "⚠️ STL not available (install statsmodels)"
    
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    if filtered.empty:
        return None, "No data available for this combination"
    ts = pd.Series(filtered['quantityKwh'].values, index=filtered['startTime_parsed']).ffill().bfill()
    
    # Sjekk for tilstrekkelig data etter fylling
    if len(ts) < 2 * period:
         return None, f"⚠️ For lite data ({len(ts)} punkter) for perioden {period}. Velg en kortere periode eller mer data."
         
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()

    # Opprett Plotly Subplots
    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, 
                           subplot_titles=("Original Series", "Trend", "Seasonal", "Residual"))

    # 1. Original Series
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Original', line=dict(color='black', width=1)), row=1, col=1)

    # 2. Trend
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend.values, mode='lines', name='Trend', line=dict(color='blue')), row=2, col=1)
    
    # 3. Seasonal
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal.values, mode='lines', name='Seasonal', line=dict(color='green')), row=3, col=1)
    
    # 4. Residual
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid.values, mode='lines', name='Residual', line=dict(color='red')), row=4, col=1)

    # Oppdater layout
    fig.update_layout(
        title_text=f"STL Decomposition: {production_group} — {price_area}",
        height=700,
        showlegend=False,
        margin=dict(t=50)
    )
    
    # Fjerne y-aksenavn (de er allerede i undertitlene) og legge til rutenett
    for i in range(1, 5):
        fig.update_yaxes(title_text="", row=i, col=1, gridcolor='lightgray')

    fig.update_xaxes(title_text="Time", row=4, col=1)
    
    return fig, None


# --- Spectrogram (Plotly) ---
def spectrogram_analysis(df, price_area, production_group, window_length=168, window_overlap=84):
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    if filtered.empty:
        return None, "No data available for this combination"
    
    production_series = pd.Series(filtered['quantityKwh'].values, index=filtered['startTime_parsed']).ffill().bfill()
    production = production_series.values
    
    if len(production) < window_length:
        return None, f"⚠️ For lite data ({len(production)} punkter) for vinduslengden {window_length}. Velg kortere vindu eller mer data."

    if _SCIPY_AVAILABLE:
        # Fs=1.0 siden vi antar en prøvetakingsfrekvens på 1 time (én prøve per time)
        f, t, Sxx = signal.spectrogram(production, fs=1.0, window='hann',
                                       nperseg=window_length, noverlap=window_overlap)
    else:
        return None, "⚠️ SciPy not available (install scipy)"
        
    Sxx_db = 10 * np.log10(Sxx + 1e-10) # Konverter til dB

    # Lag en liste over tidspunkter for x-aksen 
    # Tidene i 't' tilsvarer midtpunktet av hvert vindu, vi bruker tilsvarende indekser fra originalserien
    start_index = window_length // 2
    time_indices = production_series.index[start_index::(window_length - window_overlap)]
    
    # Sikkerhetssjekk for å matche dimensjoner
    if len(time_indices) > len(t):
        time_indices = time_indices[:len(t)]
    elif len(t) > len(time_indices):
        Sxx_db = Sxx_db[:, :len(time_indices)]
        
    # Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db, 
        x=time_indices, 
        y=f, 
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)')
    ))

    fig.update_layout(
        title=f"Spectrogram: {production_group} — {price_area}",
        xaxis_title="Time", # Endret fra 'Time (hours)' til 'Time' da x-aksen er datostempler
        yaxis_title="Frequency (cycles/hour)",
        height=600
    )
    
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
                    st.plotly_chart(fig, use_container_width=True) # Plotly-kall

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
                    st.plotly_chart(fig, use_container_width=True) # Plotly-kall

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)