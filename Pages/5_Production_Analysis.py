import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import StreamlitApplication.Data_loader as load_data    

# Page configuration
st.set_page_config(
    page_title="🌤️ Weather Data Analysis",
    layout="wide"
)

# ---------------------------
# Helpers
# ---------------------------
def get_weather_data():
    """Get weather data from session state if available, otherwise fallback to loader."""
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        return st.session_state.weather_data
    try:
        return load_data()
    except Exception:
        return None


def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there is a 'time' column of datetime64[ns]."""
    df = df.copy()

    # Case 1: Already has 'time'
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        return df

    # Case 2: Try common timestamp-like names
    for candidate in ['timestamp', 'datetime', 'date', 'period_start', 'valid_time', 'startTime', 'startTime_parsed']:
        if candidate in df.columns:
            df['time'] = pd.to_datetime(df[candidate], errors='coerce', utc=True)
            return df

    # Case 3: Maybe index is datetime
    if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        return df

    st.warning("⚠️ No 'time' or timestamp-like column found in DataFrame.")
    return df


def get_first_month_data(df: pd.DataFrame, column: str, max_points: int = 31 * 24):
    """Get data for the first month (up to max_points) for the specified column."""
    df = df.copy()
    df = ensure_time_column(df)

    if 'time' not in df.columns:
        st.error("Cannot extract first month data: No 'time' column found.")
        return pd.DataFrame(columns=['time', column])

    df = df.sort_values('time')
    start_time = df['time'].min()
    end_time = start_time + pd.Timedelta(days=31)

    monthly_data = df[(df['time'] >= start_time) & (df['time'] < end_time)][['time', column]]

    # Limit to max_points
    if len(monthly_data) > max_points:
        monthly_data = monthly_data.head(max_points)

    return monthly_data


# ---------------------------
# Main Page
# ---------------------------
st.title("🌤️ Interactive Weather Data Plot")
st.markdown("""
Explore the Data with Custom Visualizations  
Use the controls below to customize your view of the weather data.
""")

data = get_weather_data()

if data is None or len(data) == 0:
    st.warning("⚠️ No weather data loaded. Please visit the weather download page first.")
    st.info("Once you download data there, it will be available here for viewing.")
else:
    # ✅ Ensure time column
    data = ensure_time_column(data)

    # ✅ Derive year-month for filtering
    data['year_month'] = data['time'].dt.to_period('M').astype(str)

    # ✅ Rename columns for clarity (your requested names)
    rename_map = {
        'temperature_2m': 'Temperature (°C)',
        'precipitation': 'Precipitation (mm)',
        'wind_speed_10m': 'Wind Speed (m/s)',
        'wind_gusts_10m': 'Wind Gusts (m/s)',
        'wind_direction_10m': 'Wind Direction (°)'
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    st.success(f"✅ Weather data loaded: {len(data):,} records")
    # Provide a multi-select that defaults to all numeric variables and render the plot/stats here,
    # then stop further execution so the later duplicate widgets aren't shown.
    data_columns = [col for col in data.columns if col not in ['time', 'year_month'] and np.issubdtype(data[col].dtype, np.number)]

    if len(data_columns) == 0:
        st.warning("No numeric variables available to plot.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        # Multi-select defaulting to all variables
        selected_columns = st.multiselect("Select Variable to Plot", data_columns, default=data_columns)
    with col2:
        available_months = sorted(data['year_month'].unique())
        month_range = st.select_slider("Select Month Range", available_months, value=(available_months[0], available_months[-1]))

    # Filter by month range
    df_filtered = data[(data['year_month'] >= month_range[0]) & (data['year_month'] <= month_range[1])]

    # Plot selected columns (multiple allowed)
    st.subheader("📈 Weather Data Visualization")
    plt.figure(figsize=(10, 4))
    for i, col_name in enumerate(selected_columns):
        plt.plot(df_filtered['time'], df_filtered[col_name], label=col_name, linewidth=1)
    plt.title(f"{' ,'.join(selected_columns)} over Time ({month_range[0]} → {month_range[1]})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    if len(selected_columns) > 1:
        plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    st.pyplot(plt)

    # Statistics for selected columns
    st.markdown("---")
    st.subheader("📊 Basic Statistics")
    st.dataframe(df_filtered[selected_columns].describe())

    # Stop further execution to avoid duplicate widgets/plots later in the file
    st.stop()
    if 'selected_area' in st.session_state:
        sel_city = st.session_state.get('selected_city', '')
        st.info(f"📍 Data for: **{st.session_state.selected_area}** ({sel_city})")

    try:
        # Filter numeric columns
        data_columns = [col for col in data.columns if col not in ['time', 'year_month'] and np.issubdtype(data[col].dtype, np.number)]

        col1, col2 = st.columns(2)
        with col1:
            selected_column = st.selectbox("Select Variable to Plot", data_columns)
        with col2:
            available_months = sorted(data['year_month'].unique())
            month_range = st.select_slider("Select Month Range", available_months, value=(available_months[0], available_months[-1]))

        # Filter by month range
        df_filtered = data[(data['year_month'] >= month_range[0]) & (data['year_month'] <= month_range[1])]

        # ✅ Plot
        st.subheader("📈 Weather Data Visualization")
        plt.figure(figsize=(10, 4))
        plt.plot(df_filtered['time'], df_filtered[selected_column], linewidth=1)
        plt.title(f"{selected_column} over Time ({month_range[0]} → {month_range[1]})")
        plt.xlabel("Time")
        plt.ylabel(selected_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

        # ✅ Statistics
        st.markdown("---")
        st.subheader("📊 Basic Statistics")
        st.dataframe(df_filtered[[selected_column]].describe())

    except Exception as e:
        st.error(f"Error displaying data statistics: {e}")
