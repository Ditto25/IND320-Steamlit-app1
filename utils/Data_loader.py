import requests
import streamlit as st
import pandas as pd
import pymongo
from pymongo import  MongoClient
import json
import logging
from pathlib import Path
from pymongo.server_api import ServerApi


# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# --- GEO HELPERS (Necessary for weather data) ---

def area_to_geoplacement(area):
    """Converts price area (NO1-NO5) to geographical coordinates."""
    geo_dict = {
        'NO1': {'long': 10.7461, 'lat': 59.9127}, # Oslo
        'NO2': {'long': 7.9956, 'lat': 58.1467}, # Kristiansand
        'NO3': {'long': 5.3242, 'lat': 60.393},  # Bergen
        'NO4': {'long': 18.9551, 'lat': 69.6489},# Tromsø
        'NO5': {'long': 10.3951, 'lat': 63.4305}  # Trondheim
    }
    # Return standard coordinates (NO1) if the area is not found
    return geo_dict.get(area, geo_dict['NO1'])['long'], geo_dict.get(area, geo_dict['NO1'])['lat']

WEATHER_AREAS = {
    'NO1': {'city': 'Oslo', 'latitude': 59.9139, 'longitude': 10.7522},
    'NO2': {'city': 'Kristiansand', 'latitude': 58.1462, 'longitude': 7.9956},
    'NO3': {'city': 'Trondheim', 'latitude': 63.4305, 'longitude': 10.3951},
    'NO4': {'city': 'Tromsø', 'latitude': 69.6492, 'longitude': 18.9553},
    'NO5': {'city': 'Bergen', 'latitude': 60.3913, 'longitude': 5.3221}
}


@st.cache_data(ttl=3600, show_spinner="Downloading hourly weather data...")
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

# Helper function to download and store weather data in session state
def download_and_store_weather_data(lon, lat, area, city, year):
    with st.spinner(f"Downloading hourly data for {city} ({year})..."):
        try:
            weather_data = download_hourly_weather_data(
                longitude=lon,
                latitude=lat,
                year=year
            )
            st.session_state.weather_data = weather_data
            st.session_state.selected_area = area
            st.session_state.selected_city = city
            st.session_state.selected_year = year
            st.success(f"✅ Downloaded {len(weather_data):,} records for {area} ({year})!")
            
        except Exception as e:
            st.error(f"Error downloading weather data: {e}")

def render_weather_selector(key_suffix=""):
    """
    Renders price area and year selector in the Streamlit sidebar, 
    and displays status of the currently loaded weather data.
    """
    with st.sidebar:
        st.subheader("📍 Data Selector")
        price_areas = list(WEATHER_AREAS.keys())
        
        # Area Selection (Select Box)
        selected_area = st.selectbox(
            "Price Area",
            options=price_areas,
            format_func=lambda x: f"{x} - {WEATHER_AREAS[x]['city']}",
            key="sidebar_selected_area",
            help="Select the geographical price area for which weather data will be downloaded."
        )
        
        # Year Selection (Select Box)
        selected_year = st.selectbox(
            "Year",
            options=list(range(2024, 2009, -1)), 
            index=1, 
            key=f"sidebar_selected_year_{key_suffix}"
        )
        
        # Get coordinates for the selected area
        selected_city = WEATHER_AREAS[selected_area]['city']
        selected_lat = WEATHER_AREAS[selected_area]['latitude']
        selected_lon = WEATHER_AREAS[selected_area]['longitude']

        st.caption(f"Location: {selected_city} ({selected_year})")
        
        # Download Button
        if st.button("Download Weather Data", key="sidebar_download_button"):
            download_and_store_weather_data(selected_lon, selected_lat, selected_area, selected_city, selected_year)

        st.markdown("---") 

        # Loaded Data Status
        st.subheader("Loaded Data Status")
        
        if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
            df = st.session_state.weather_data
            num_rows = len(df)
            
            # Retrieve stored metadata
            loaded_area = st.session_state.get('selected_area', 'Unknown')
            loaded_city = st.session_state.get('selected_city', 'Unknown')
            loaded_year = st.session_state.get('selected_year', 'Unknown')
            
            st.success(f"✅ Data Loaded!")
            st.caption(f"**Area:** {loaded_area} - {loaded_city}")
            st.caption(f"**Year:** {loaded_year}")
            st.caption(f"**Hourly Points:** {num_rows:,}")
            
            # Button to clear the data
            if st.button("Clear Weather Data", key="clear_weather_data"):
                st.session_state.weather_data = None
                st.session_state.selected_area = None
                st.session_state.selected_city = None
                st.session_state.selected_year = None
                st.rerun()
                
        else:
            st.warning("⚠️ No weather data loaded.")
    

# Load and process data
@st.cache_data(ttl=3600, show_spinner="Loading data from MongoDB...")
def load_data():
    """Load and process data from MongoDB"""
    client = get_mongo_client()
    
    database = client['Database'] 
    collection = database['data']
    
    # Fetch all documents from MongoDB
    records = list(collection.find({}, {'_id': 0}))
    
    if not records:
        st.error("No data found in MongoDB! Please run your notebook to insert data first.")
        st.stop()
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Clean the data - remove any records with list or invalid values
    def is_valid_record(row):
        """Check if a record has valid data types"""
        for col in ['startTime', 'endTime', 'lastUpdatedTime', 'priceArea', 'productionGroup', 'quantityKwh']:
            if col in row and isinstance(row[col], list):
                return False
        return True
    
    # Filter out invalid records
    valid_indices = df.apply(is_valid_record, axis=1)
    initial_count = len(df)
    df = df[valid_indices].reset_index(drop=True)
    
    if len(df) < initial_count:
        st.warning(f"Filtered out {initial_count - len(df)} invalid records from the dataset.")
    
    # Convert date columns to datetime (with error handling)
    try:
        df['startTime'] = pd.to_datetime(df['startTime'], errors='coerce')
        df['endTime'] = pd.to_datetime(df['endTime'], errors='coerce')
        df['lastUpdatedTime'] = pd.to_datetime(df['lastUpdatedTime'], errors='coerce')
        
        # Remove rows where datetime conversion failed
        df = df.dropna(subset=['startTime']).reset_index(drop=True)
        
        # Add month columns
        df['month'] = df['startTime'].dt.month
        df['month_name'] = df['startTime'].dt.strftime('%B')
        
    except Exception as e:
        st.error(f"Error processing datetime columns: {e}")
        st.stop()
    
    return df
# MongoDB connection
@st.cache_resource(ttl=3600, show_spinner="Connecting to MongoDB...")
def get_mongo_client():
    """Create and return MongoDB client"""
    db_user = st.secrets["database"]["db_user"]
    secret = st.secrets["database"]["secret"]

    uri = f"mongodb+srv://{db_user}:{secret}@cluster1.g046i3t.mongodb.net/?appName=Cluster1"
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    # Test connection
    try:
        client.admin.command('ping')
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
    
    return client

# Cache function for loading consumption data from MongoDB (power / consumption_2021_2024_hourly)
# I utils/Data_loader.py

@st.cache_data(ttl=3600, show_spinner="Fetching consumption data from MongoDB...")
def load_elhub_consumption_data() -> pd.DataFrame:
    """
    Load consumption_2021_2024_hourly from MongoDB (all years).
    """
    try:
        client = MongoClient(st.secrets["MONGO"]["uri"])
        db = client["power"]
        col = db["consumption_2021_2024_hourly"]
        
        # Hent data
        records = list(col.find({}, {"_id": 0}))
    except Exception as e:
        st.error(f"Feil ved kobling til database: {e}")
        return pd.DataFrame()
    finally:
        client.close()

    # Håndter tomt datasett
    if not records:
        # Returner en tom DataFrame med forventede kolonner for å unngå KeyError senere
        return pd.DataFrame(columns=["startTime", "consumptionGroup", "priceArea", "quantityKwh"])

    df = pd.DataFrame(records)
    
    # Sørg for at startTime konverteres hvis den finnes
    if "startTime" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"])
        
    return df

# I filen: utils/Data_loader.py

@st.cache_data(ttl=3600, show_spinner="Fetching consumption data from MongoDB...")
def load_elhub_consumption_data() -> pd.DataFrame:
    """
    Load consumption_2021_2024_hourly from MongoDB.
    """
    try:
        client = MongoClient(st.secrets["MONGO"]["uri"])
        db = client["power"]
        col = db["consumption_2021_2024_hourly"]
        records = list(col.find({}, {"_id": 0}))
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        # Det er god praksis å lukke klienten, men pass på at st.cache_data håndterer dette greit.
        # Hvis du får problemer med 'client closed', kan du fjerne client.close()
        client.close()

    # --- SJEKK OM DATASETTET ER TOMT ---
    if not records:
        # Returner en tom DF med de forventede kolonnene for å unngå KeyError i Map-appen
        return pd.DataFrame(columns=["startTime", "consumptionGroup", "priceArea", "quantityKwh"])

    df = pd.DataFrame(records)

    # Sikkerhetskonvertering: Kun hvis kolonnen finnes
    if "startTime" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"])
    
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching production data from MongoDB...")
def load_elhub_production_data() -> pd.DataFrame:
    """
    Load production_2022_2024_hourly from MongoDB.
    """
    try:
        client = MongoClient(st.secrets["MONGO"]["uri"])
        db = client["power"]
        
        col = db["production_2022_2024_hourly"] 

        records = list(col.find({}, {"_id": 0}))
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        client.close()
        
    # Håndtering av tomt datasett (som vi fikset i forrige steg)
    if not records:
        return pd.DataFrame(columns=["startTime", "productionGroup", "priceArea", "quantityKwh"])

    df = pd.DataFrame(records)
    
    if "startTime" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"])
        df["year"] = df["startTime"].dt.year

    return df


# Cache function for loading Open-Meteo data from the API
@st.cache_data(ttl=3600, show_spinner="Loading Open-Meteo data from API...")
def load_open_meteo_api(
    latitude: float,
    longitude: float,
    year: int = 2022, 
    area: str | None = None,
) -> pd.DataFrame:
    """Download hourly weather data for given coordinates and year."""
    url = "https://archive-api.open-meteo.com/v1/archive"

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "models": "era5",
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

    # Cache function for loading price area GeoJSON file
@st.cache_data(ttl=86400, show_spinner="Loading price area GeoJSON...")
def load_pricearea_geojson() -> dict:
    """
    Load the price area GeoJSON file from data/file.geojson.
    """
    geo_path = Path(__file__).parent.parent / "data" / "file.geojson"
    with geo_path.open("r", encoding="utf-8") as f:
        return json.load(f)
