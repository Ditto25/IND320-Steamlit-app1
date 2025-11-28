import streamlit as st
import pandas as pd
from utils.Data_loader import download_weather_data, render_sidebar_info 

# --- NYTT: Renderer styling og statusinfo i sidemenyen ---
render_sidebar_info()

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


# Main page content (customized text)
st.markdown("📅 Choose a price area and year to download hourly weather data")

st.markdown("---")

# Function to download hourly weather data
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
                # KORRIGERT KALL: Bruker importert funksjon og kartlegger år til start_year/end_year
                weather_data = download_weather_data(
                    latitude=selected_lat,
                    longitude=selected_lon,
                    start_year=selected_year,
                    end_year=selected_year
                )

                # Lagrer data i session state
                st.session_state.weather_data = weather_data
                st.session_state.selected_area = selected_area
                st.session_state.selected_city = selected_city
                st.session_state.selected_year = selected_year # Oppdaterer til det faktiske valgte året

                st.success(f"Successfully downloaded {len(weather_data):,} hourly records for {selected_city} ({selected_year}).")

                st.markdown("---")
                st.subheader("Data Summary 📊")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Hours", f"{len(weather_data):,}")
                with col2:
                    # KORRIGERT: Bruker kolonnenavnet 'time' som er definert i Data_loader
                    st.metric("Start Timestamp", weather_data['time'].min().strftime('%Y-%m-%d %H:%M'))
                with col3:
                    # KORRIGERT: Bruker kolonnenavnet 'time' som er definert i Data_loader
                    st.metric("End Timestamp", weather_data['time'].max().strftime('%Y-%m-%d %H:%M'))

                with st.expander("Preview (first 48 rows)"):
                    preview = weather_data.head(48).copy()
                    # KORRIGERT: Bruker kolonnenavnet 'time' som er definert i Data_loader
                    preview['time'] = preview['time'].dt.strftime('%Y-%m-%d %H:%M') 
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