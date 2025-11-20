import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor

# Page configuration
st.set_page_config(
    page_title="Weather Anomalies Detection",
    layout="wide"
)

# Load weather data from session state (set by page 2)
def get_weather_data():
    """Get weather data from session state"""
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        return st.session_state.weather_data
    return None

def detect_temperature_outliers(df, freq_cutoff=0.05, n_std=3):
    """
    Detect temperature outliers using DCT and SPC
    """
    # Extract temperature data
    temp = df['temperature (°C)'].ffill().bfill().values
    time = pd.to_datetime(df['time'])
    
    # Apply DCT
    temp_dct = dct(temp, type=2, norm='ortho')
    
    # High-pass filter
    cutoff_index = int(len(temp_dct) * freq_cutoff)
    temp_dct_filtered = temp_dct.copy()
    temp_dct_filtered[:cutoff_index] = 0
    
    # Get SATV
    satv = idct(temp_dct_filtered, type=2, norm='ortho')
    
    # Calculate robust statistics
    median_satv = np.median(satv)
    mad_satv = np.median(np.abs(satv - median_satv))
    std_satv = mad_satv * 1.4826
    
    # SPC boundaries
    upper_boundary = median_satv + n_std * std_satv
    lower_boundary = median_satv - n_std * std_satv
    
    # Identify outliers
    outliers_mask = (satv > upper_boundary) | (satv < lower_boundary)
    n_outliers = np.sum(outliers_mask)
    outlier_percentage = (n_outliers / len(temp)) * 100
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot normal points
    ax.plot(time[~outliers_mask], temp[~outliers_mask], 
            color='blue', linewidth=1, alpha=0.7, label='Normal temperature')
    
    # Plot outliers
    ax.plot(time[outliers_mask], temp[outliers_mask], 
            'o', color='red', markersize=4, alpha=0.8, label=f'Outliers (n={n_outliers})')
    
    # Plot boundaries
    ax.axhline(y=np.mean(temp) + upper_boundary, color='orange', 
               linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Upper boundary (+{n_std}σ)')
    ax.axhline(y=np.mean(temp) + lower_boundary, color='orange', 
               linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Lower boundary (-{n_std}σ)')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Temperature Outlier Detection using DCT and SPC\n' +
                 f'Frequency cutoff: {freq_cutoff}, SPC boundaries: ±{n_std}σ',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    results = {
        'figure': fig,
        'n_outliers': n_outliers,
        'outlier_percentage': outlier_percentage,
        'upper_boundary': upper_boundary,
        'lower_boundary': lower_boundary,
        'outlier_dates': time[outliers_mask].tolist(),
        'outlier_values': temp[outliers_mask].tolist()
    }
    
    return results

def detect_precipitation_anomalies(df, outlier_proportion=0.01, n_neighbors=50):
    """
    Detect precipitation anomalies using LOF
    """
    # Extract precipitation data
    precip = df['precipitation (mm)'].values
    time = pd.to_datetime(df['time'])
    
    # Prepare features for LOF
    precip_diff = np.diff(precip, prepend=precip[0])
    X = np.column_stack([precip, precip_diff])
    
    # Add small jitter to handle duplicate values (common with zero precipitation)
    jitter = np.random.RandomState(42).normal(0, 1e-6, X.shape)
    X_jittered = X + jitter
    
    # Fit LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=outlier_proportion)
    predictions = lof.fit_predict(X_jittered)
    lof_scores = lof.negative_outlier_factor_
    
    # Identify anomalies
    anomalies_mask = predictions == -1
    n_anomalies = np.sum(anomalies_mask)
    anomaly_percentage = (n_anomalies / len(precip)) * 100
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Convert to proper datetime format
    time_numeric = time.to_numpy()
    
    # Plot normal precipitation
    ax.bar(time_numeric[~anomalies_mask], precip[~anomalies_mask], 
           width=0.04, color='blue', alpha=0.6, label='Normal precipitation')
    
    # Plot anomalies
    ax.bar(time_numeric[anomalies_mask], precip[anomalies_mask], 
           width=0.04, color='red', alpha=0.8, label=f'Anomalies (n={n_anomalies})')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Precipitation Anomaly Detection using Local Outlier Factor (LOF)\n' +
                 f'Expected outlier proportion: {outlier_proportion*100:.1f}%',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Get anomaly details
    anomaly_dates_list = time[anomalies_mask].tolist()
    
    results = {
        'figure': fig,
        'n_anomalies': n_anomalies,
        'anomaly_percentage': anomaly_percentage,
        'lof_scores': lof_scores,
        'anomaly_dates': anomaly_dates_list,
        'anomaly_values': precip[anomalies_mask].tolist()
    }
    
    return results

# Main page content
st.title("Weather Anomalies Detection")
st.header("Temperature Outliers and Precipitation Anomalies")

st.markdown("""
Detect unusual weather patterns using advanced statistical methods.
""")

st.markdown("---")

# Check if weather data is available
weather_df = get_weather_data()
# --- Normalize time column naming ---
def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        return df
    for candidate in ['timestamp', 'datetime', 'date', 'period_start', 'valid_time']:
        if candidate in df.columns:
            df['time'] = pd.to_datetime(df[candidate], errors='coerce', utc=True)
            return df
    if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        return df
    st.warning("⚠️ No 'time' or timestamp-like column found in DataFrame.")
    return df

# --- Apply the fix right after loading the data ---
if weather_df is not None:
    weather_df = ensure_time_column(weather_df)


if weather_df is None:
    st.warning("⚠️ No weather data loaded. Please visit the **Price Area Selector** page first to download weather data.")
    st.info("The weather data will be downloaded based on your selected price area and will be available for analysis on this page.")
else:
    st.success(f"✅ Weather data loaded: {len(weather_df)} records from {weather_df['time'].min().date()} to {weather_df['time'].max().date()}")
    
    # Display selected area info
    if 'selected_area' in st.session_state:
        st.info(f"📍 Analyzing data for: **{st.session_state.selected_area}** ({st.session_state.selected_city})")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🌡️ Temperature Outliers (SPC)", "🌧️ Precipitation Anomalies (LOF)"])
    
    # Tab 1: Temperature Outliers
    with tab1:
        st.subheader("Temperature Outlier Detection using DCT and SPC")
        
        st.markdown("""
        This method uses:
        - **DCT (Direct Cosine Transform)**: Removes seasonal trends via high-pass filtering
        - **SATV (Seasonally Adjusted Temperature Variations)**: Detrended temperature data
        - **SPC (Statistical Process Control)**: Identifies outliers using robust statistics
        """)
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            freq_cutoff = st.slider(
                "Frequency Cutoff",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Lower = removes more seasonal trends (0.05 recommended)"
            )
        
        with col2:
            n_std = st.slider(
                "Standard Deviations",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Number of σ for SPC boundaries (3σ = 99.7%)"
            )
        
        # Perform analysis
        if st.button("Detect Temperature Outliers", key="temp_button"):
            with st.spinner("Analyzing temperature data..."):
                results = detect_temperature_outliers(
                    weather_df,
                    freq_cutoff=freq_cutoff,
                    n_std=n_std
                )
                
                st.pyplot(results['figure'])
                
                # Statistics
                st.markdown("---")
                st.subheader("📊 Outlier Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Outliers", results['n_outliers'])
                with col2:
                    st.metric("Percentage", f"{results['outlier_percentage']:.2f}%")
                with col3:
                    st.metric("Boundary Range", f"±{n_std}σ")
                
                # Show outlier details
                if results['n_outliers'] > 0:
                    with st.expander("🔍 View Outlier Details"):
                        outlier_df = pd.DataFrame({
                            'Date': [d.strftime('%Y-%m-%d %H:%M:%S') if hasattr(d, 'strftime') else str(d) for d in results['outlier_dates']],
                            'Temperature (°C)': [f"{v:.2f}" for v in results['outlier_values']]
                        })
                        st.dataframe(outlier_df, width='stretch', height=300)
                
                st.info("""
                **Interpretation:**
                - Points outside the orange boundaries are classified as outliers
                - These represent unusual temperature deviations from seasonal norms
                - Lower frequency cutoff = more aggressive detrending
                """)
    
    # Tab 2: Precipitation Anomalies
    with tab2:
        st.subheader("Precipitation Anomaly Detection using LOF")
        
        st.markdown("""
        **Local Outlier Factor (LOF)** detects anomalies based on local density:
        - Points with significantly lower density than neighbors are anomalies
        - Well-suited for precipitation data with extreme values
        - Considers both value and rate of change
        """)
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            outlier_prop = st.slider(
                "Expected Outlier Proportion",
                min_value=0.001,
                max_value=0.05,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Expected proportion of anomalies (1% = 0.01)"
            )
        
        with col2:
            n_neighbors = st.slider(
                "Number of Neighbors",
                min_value=20,
                max_value=200,
                value=50,
                step=10,
                help="Higher values reduce impact of duplicate values"
            )
        
        st.info(f"Expecting approximately {int(len(weather_df) * outlier_prop)} anomalies out of {len(weather_df)} data points")
        
        # Perform analysis
        if st.button("Detect Precipitation Anomalies", key="precip_button"):
            with st.spinner("Analyzing precipitation data..."):
                results = detect_precipitation_anomalies(
                    weather_df,
                    outlier_proportion=outlier_prop,
                    n_neighbors=n_neighbors
                )
                
                st.pyplot(results['figure'])
                
                # Statistics
                st.markdown("---")
                st.subheader("📊 Anomaly Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", results['n_anomalies'])
                with col2:
                    st.metric("Percentage", f"{results['anomaly_percentage']:.2f}%")
                with col3:
                    st.metric("Mean LOF Score", f"{np.mean(results['lof_scores']):.2f}")
                
                # Show anomaly details
                if results['n_anomalies'] > 0:
                    with st.expander("🔍 View Anomaly Details (Top 20)"):
                        # Sort by precipitation value
                        anomaly_df = pd.DataFrame({
                            'Date': [d.strftime('%Y-%m-%d %H:%M:%S') if hasattr(d, 'strftime') else str(d) for d in results['anomaly_dates']],
                            'Precipitation (mm)': results['anomaly_values']
                        })
                        anomaly_df = anomaly_df.sort_values('Precipitation (mm)', ascending=False)
                        st.dataframe(anomaly_df.head(20), width='stretch', height=400)
                
                st.info("""
                **Interpretation:**
                - Red bars indicate anomalous precipitation events
                - LOF identifies patterns that deviate from local neighborhood density
                - More negative LOF scores = more anomalous
                """)
    
