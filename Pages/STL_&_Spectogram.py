from unittest import result
import streamlit as st
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import toml
import numpy as np
from functools import lru_cache
import plotly.graph_objs as go
import plotly.subplots as sp
from utils.Data_loader import WEATHER_AREAS, render_weather_selector, load_data




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

st.set_page_config(page_title="STL & Spectrogram: Advanced Time Series Analysis", layout="wide")
render_weather_selector()

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

# --- STL Decomposition (Plotly) ---
def stl_analysis(df, price_area, production_group, period=24, seasonal=7, trend=None, robust=False):
    if not _STL_AVAILABLE:
        return None, "⚠️ STL not available (install statsmodels)"
    
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    if filtered.empty:
        return None, "No data available for this combination"
    ts = pd.Series(filtered['quantityKwh'].values, index=filtered['startTime']).ffill().bfill()
    
    # Sjekk for tilstrekkelig data etter fylling
    if len(ts) < 2 * period:
         return None, f"⚠️ For lite data ({len(ts)} punkter) for perioden {period}. Velg en kortere periode eller mer data."
         
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()
    # 1. Residual Volatility (Standard Deviation of Residuals)
    residual_std = result.resid.std()
    
    # 2. Seasonal Strength Index (Ratio of Seasonal variance to Residual variance)
    # Legg til en liten verdi (1e-6) for å unngå divisjon med null hvis variansen er 0
    seasonal_strength = result.seasonal.var() / (result.resid.var() + 1e-6) 
    
    # Variansen til hver komponent
    var_seasonal = result.seasonal.var()
    var_trend = result.trend.var()
    var_residual = result.resid.var()

    # Total varians for å beregne relative andeler
    var_total = var_seasonal + var_trend + var_residual

    # Beregn relative andeler (for å unngå divisjon med null, sjekk total varians)
    if var_total > 1e-6:
        # Prosentandel av total varians forklart av hver komponent
        percent_seasonal = (var_seasonal / var_total) * 100
        percent_trend = (var_trend / var_total) * 100
        percent_residual = (var_residual / var_total) * 100
    else:
        # Hvis dataen er flat eller mangler, sett til null
        percent_seasonal, percent_trend, percent_residual = 0, 0, 0
    
    # 3. Long-Term Trend Change (Difference between end and start of trend component)
    if not result.trend.empty:
        trend_change = result.trend.iloc[-1] - result.trend.iloc[0]
    else:
        trend_change = 0
    # Opprett Plotly Subplots
    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, 
                           subplot_titles=("Original Series", "Trend", "Seasonal", "Residual"))

    # 1. Original Series
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Original', line=dict(color='orange', width=1)), row=1, col=1)

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
    
    return fig, None, residual_std, seasonal_strength, trend_change, percent_seasonal, percent_trend, percent_residual
# --- Spectrogram (Plotly) ---
def spectrogram_analysis(df, price_area, production_group, window_length=168, window_overlap=84):
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    if filtered.empty:
        return None, "No data available for this combination"
    
    production_series = pd.Series(filtered['quantityKwh'].values, index=filtered['startTime']).ffill().bfill()
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
    df = load_data()
    price_areas = sorted(df['priceArea'].unique())
    production_groups = sorted(df['productionGroup'].unique())

    tab1, tab2 = st.tabs(["🧩 Seasonal-Trend Decomposition", "🎵 Spectrogram"])

    default_area_key = st.session_state.get('sidebar_selected_area')
    try:
        default_index = price_areas.index(default_area_key) if default_area_key in price_areas else 0
    except Exception:
        default_index = 0
        
    with tab1:
        col1, col2, col3 = st.columns(3)
        stl_area = col1.selectbox(
            "Price Area", 
            price_areas, 
            index=default_index, 
            format_func=lambda x: f"{x} - {WEATHER_AREAS[x]['city']}", 
            key="stl_price_area"
        )
        stl_group = col2.selectbox("Production Group", production_groups, key="stl_production_group")
        stl_period = col3.number_input("Seasonal Period", 2, 720, 24, help="Defines the length of the seasonal cycle (e.g., 24 for hourly daily cycle, 168 for weekly cycle).")
        col4,col5 = st.columns(2)
        stl_seasonal = col4.slider("Seasonal Smoothing", 3, 25, 7, step=2, help="Controls the smoothness of the Seasonal component. Lower values (e.g., 3) capture more detail; higher values (e.g., 19) produce a smoother curve.")
        stl_robust = col5.checkbox("Robust Fitting", True, help="If checked, the analysis is robust to outliers (extreme values). Outliers are given less weight, preventing them from distorting the overall Trend and Seasonal components.")

        if st.button("Run STL Analysis", key="Run_stl_button"):
            with st.spinner("Running STL Decomposition..."):
                filtered_df = df[(df['priceArea'] == stl_area) & (df['productionGroup'] == stl_group)]
                                
                fig, error, res_std, s_strength, t_change, p_seasonal, p_trend, p_residual = stl_analysis(
                    filtered_df, 
                    stl_area, 
                    stl_group, 
                    stl_period, 
                    stl_seasonal, 
                    robust=stl_robust
                )
                if error:
                    st.error(error)
                else:
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.subheader("📊 Component Summary (Variance Contribution)")

                    # Vis prosentandelene i en egen rad
                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        label="Seasonal Contribution", 
                        value=f"{p_seasonal:.1f} %", 
                        help="Percentage of the total time series variance explained by the repeating Seasonal pattern."
                    )
                    c2.metric(
                        label="Trend Contribution", 
                        value=f"{p_trend:.1f} %", 
                        help="Percentage of the total time series variance explained by the long-term Trend."
                    )
                    c3.metric(
                        label="Residual (Noise)", 
                        value=f"{p_residual:.1f} %", 
                        help="Percentage of the total variance remaining after extracting Seasonal and Trend components (unexplained variation)."
                    ) 

                    st.markdown("---") 

                    # Use st.expander to hide the detailed explanation
                    with st.expander("🔍 Interpretation: What do the Variance Contributions mean?"):
                        st.markdown("""
                        ### 🟢 Seasonal Contribution
                        * **High Value (e.g., > 30%):** Indicates the data is strongly dominated by **periodic patterns** (daily/weekly cycles). Energy consumption often shows this.
                        * **Low Value (e.g., < 5%):** Indicates that the seasonal pattern is weak or irrelevant compared to the large, long-term changes. This is typical for constant production or short time spans.
                        ### 🔵 Trend Contribution
                        * **High Value (e.g., > 50%):** Indicates that **long-term changes** (increases/decreases over months) have the largest impact on the total fluctuation. This can be due to natural shifts (summer/winter) or structural changes (new installed capacity).
                        ### 🔴 Residual (Noise)
                        * **High Value (e.g., > 30%):** Means a large portion of the variation is **unexplained noise** and does not follow a fixed pattern. This is often caused by unpredictable events like extreme weather, operational faults, or a lack of important variables (like prices) in the model.
                        * **Low Value (e.g., < 10%):** Means the model is strong; almost all variation is captured by the Trend and Seasonal components.
                        """)
    with tab2:
        # Main expander for the Spectrogram tools and results
        #             # --- Analysis Title and Description ---
            st.header("Spectrogram Analysis (Frequency over Time)")
            st.markdown("""
                The Spectrogram visualizes how the **frequency content** of the time series changes over time. 
                This technique is excellent for identifying **cyclic patterns** (daily, weekly) and seeing 
                if the strength of these cycles varies (e.g., between summer and winter).
            """)

            # --- Parameter Selection ---
            col1, col2 = st.columns(2)
            spec_area = col1.selectbox("Price Area", price_areas, key="spec_price_area", 
                                    help="Select the geographical price area to analyze.")
            spec_group = col2.selectbox("Production Group", production_groups, key="spec_production_group",
                                        help="Select the specific type of energy production data to be analyzed.")

            col3, col4 = st.columns(2)
            spec_window = col3.slider(
                "Window Length (hours)", 
                24, 720, 168, step=24,
                help="The length of the time window used for each Fourier Transform. A longer window gives better frequency resolution, while a shorter window improves time resolution."
            )
            spec_overlap = col4.slider(
                "Window Overlap (hours)", 
                0, int(spec_window * 0.9), int(spec_window * 0.5), step=12,
                help="The amount the next time window overlaps with the previous one. A 50% overlap provides a smooth result."
            )
            
            # --- Analysis Button ---
            if st.button("Create Spectrogram", key="create_spec_button"):
                with st.spinner("Computing Spectrogram..."):
                    fig, error = spectrogram_analysis(df, spec_area, spec_group, spec_window, spec_overlap)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Analysis complete for {spec_group} – {spec_area}.")
                        
                        # Display the Spectrogram plot
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("---") 

                        # --- INTERPRETATION EXPANDER ---
                        with st.expander("🔍 Interpretation: What do the Spectrogram Patterns mean?"):
                            st.markdown("""
                            ### Spectrogram Interpretation: Frequency Over Time

                            This chart visualizes the **strength (color)** of different **cycles (Y-axis)** over **time (X-axis)**.

                            ---

                            #### 🟡 Strong Patterns (Yellow/Bright Colors)
                            * **Horizontal Stripes:** Indicate **constant, dominant cycles** present throughout the year.
                                * **≈ 0.04 cycles/hour:** Represents the **Daily Cycle** (24 hours).
                                * **≈ 0.006 cycles/hour:** Represents the **Weekly Cycle** (168 hours).

                            #### 🔵 Weak Patterns (Blue/Purple)
                            * **Dark Areas:** Mean there is **no significant periodicity** in that frequency band (mostly noise).

                            #### 🌊 Change Over Time (X-axis)
                            * **Color Fading Along X-axis:** Shows that the **strength** of a cycle changes seasonally.
                                * If a stripe fades (e.g., in summer), the daily/weekly pattern is **less dominant** during that period.

                            #### ⏫ Higher Frequencies
                            * **Above 0.1 cycles/hour:** Represents cycles shorter than 10 hours. These are generally **random noise** and not significant patterns.
                            """)
          
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)