import streamlit as st
import pandas as pd
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Weather Data Downloader",
    layout="wide"
)

# --- MAIN TITLE ---
st.title("🌦️ Weather Data Downloader")
st.caption("Select a Norwegian price area and year to download hourly weather data.")

st.markdown("---")


# Hardcoded Norwegian price areas / cities with coordinates
Data = {
    'NO1': {'city': 'Oslo', 'latitude': 59.9139, 'longitude': 10.7522},
    'NO2': {'city': 'Kristiansand', 'latitude': 58.1462, 'longitude': 7.9956},
    'NO3': {'city': 'Trondheim', 'latitude': 63.4305, 'longitude': 10.3951},
    'NO4': {'city': 'Tromsø', 'latitude': 69.6492, 'longitude': 18.9553},
    'NO5': {'city': 'Bergen', 'latitude': 60.3913, 'longitude': 5.3221}
}

# Download hourly weather data from open-meteo API
@st.cache_data
def download_hourly_weather_data(longitude, latitude, year):
    """
    Download hourly weather data from Open-Meteo archive (ERA5) for the given year.
    Returns a pandas DataFrame with a datetime index (timestamp) and hourly rows.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': f"{year}-01-01",
        'end_date': f"{year}-12-31",
        # request hourly variables
        'hourly': [
            'temperature_2m',
            'apparent_temperature',
            'precipitation',
            'windspeed_10m',
            'windgusts_10m',
            'winddirection_10m',
            'relativehumidity_2m'
        ],
        'timezone': 'auto'
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")

    data = response.json()
    if 'hourly' not in data:
        raise Exception("Unexpected API response: 'hourly' key not found")

    hourly = data['hourly']

    # build DataFrame with full timestamps (not just dates)
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        'temperature (°C)': hourly.get('temperature_2m'),
        'apparent_temperature (°C)': hourly.get('apparent_temperature'),
        'precipitation (mm)': hourly.get('precipitation'),
        'windspeed (m/s)': hourly.get('windspeed_10m'),
        'windgusts (m/s)': hourly.get('windgusts_10m'),
        'winddir (°)': hourly.get('winddirection_10m'),
        'relative_humidity (%)': hourly.get('relativehumidity_2m')
    })

    # optional: set timestamp as index
    df.set_index('timestamp', inplace=False)  # keep as column for display; adjust if you prefer index

    return df

# Main page content (customized text)
st.markdown("📅 Choose a price area and year to download hourly weather data")

st.markdown("---")

try:
    st.subheader("Select Price Area 📍")
    price_areas = list(Data.keys())
    selected_area = st.radio(
        "Choose a Norwegian price area:",
        options=price_areas,
        format_func=lambda x: f"{x} - {Data[x]['city']}",
        horizontal=True,
        help="Hardcoded price areas mapped to representative city coordinates"
    )

    selected_city = Data[selected_area]['city']
    selected_lat = Data[selected_area]['latitude']
    selected_lon = Data[selected_area]['longitude']

    st.info(f"Selected: {selected_area} — {selected_city} ({selected_lat:.4f}, {selected_lon:.4f})")

    st.markdown("---")
    st.subheader("Select Year (hourly data) 📅")

    selected_year = st.selectbox(
        "Choose a year for hourly weather data:",
        options=list(range(2024, 2009, -1)),  # 2010..2024
        index=1,  # default to 2023
        help="Select year between 2010 and 2024"
    )

    st.info(f"Selected year: {selected_year}")

    st.markdown("---")
    st.subheader("Download Hourly Weather Data ⬇️🌤️")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Download hourly weather data for {selected_city} for the year {selected_year}.")
    with col2:
        download_button = st.button("Download Hourly Data")

    if download_button:
        with st.spinner(f"Downloading hourly data for {selected_city} ({selected_year})..."):
            try:
                weather_data = download_hourly_weather_data(
                    longitude=selected_lon,
                    latitude=selected_lat,
                    year=selected_year
                )

                # store in session state
                st.session_state.weather_data = weather_data
                st.session_state.selected_area = selected_area
                st.session_state.selected_city = selected_city
                st.session_state.selected_year = selected_year

                st.success(f"Successfully downloaded {len(weather_data):,} hourly records for {selected_city} ({selected_year}).")

                st.markdown("---")
                st.subheader("Data Summary 📊")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Hours", f"{len(weather_data):,}")
                with col2:
                    st.metric("Start Timestamp", weather_data['timestamp'].min().strftime('%Y-%m-%d %H:%M'))
                with col3:
                    st.metric("End Timestamp", weather_data['timestamp'].max().strftime('%Y-%m-%d %H:%M'))

                with st.expander("Preview (first 48 rows)"):
                    preview = weather_data.head(48).copy()
                    preview['timestamp'] = preview['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(preview)

                with st.expander("Statistical Summary (numeric columns)"):
                    numeric_cols = weather_data.select_dtypes(include='number').columns
                    st.dataframe(weather_data[numeric_cols].describe())

            except Exception as e:
                st.error(f"Error downloading hourly weather data: {e}")
                st.exception(e)

    # If data loaded in session, show status and option to clear
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        st.markdown("---")
        st.success("Hourly weather data loaded and available for other pages.")
        loaded_year = st.session_state.get('selected_year', 'Unknown')
        st.info(f"Current data: {st.session_state.selected_area} — {st.session_state.selected_city} ({loaded_year})")

        if st.button("Clear Loaded Data"):
            st.session_state.weather_data = None
            st.session_state.selected_area = None
            st.session_state.selected_city = None
            st.session_state.selected_year = None
            st.rerun()

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
