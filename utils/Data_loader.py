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
def render_sidebar_info():
    """
    Shows application info and global settings in the sidebar.
    """
    with st.sidebar:
        # Main section for app information
        st.header("⚙️ Application Controls")
        st.markdown(
            """
            This sidebar shows information about the data and allows you to
            set global parameters.
            """
        )

        # Data loading status (Can be dynamic in a more complex app)
        st.subheader("Data Status")
        if 'production_data' in st.session_state and st.session_state.production_data is not None:
            data_points = len(st.session_state.production_data)
            start_date = st.session_state.production_data['startTime_parsed'].min().strftime('%Y-%m-%d')
            end_date = st.session_state.production_data['endTime_parsed'].max().strftime('%Y-%m-%d')
            
            st.success(f"Data loaded!")
            st.caption(f"Period: {start_date} to {end_date}")
            st.caption(f"Number of points: {data_points:,}")
        else:
             st.warning("⚠️ No energy data loaded.")

        # Example of a global input field in the sidebar
        st.subheader("Analysis Settings")
        global_sample_rate = st.selectbox(
            "Sampling Rate for Analysis",
            ["Time", "Day", "Week"],
            key="global_sample_rate_sidebar"
        )
        st.info(f"Analysis runs on a {global_sample_rate} basis.")
        
        st.markdown("---")
        st.caption("Developed for Advanced Time Series Analysis.")

# --- MAIN CLASS FOR DATA LOADING ---

    """Handles connection to MongoDB and loading of energy and weather data."""
    
    def __init__(_self):
        _self.client = _self._init_connection()
        _self.db = None
        _self.collections = {}
        if _self.client:
            _self.db = _self.client['power'] # Assumed database name
            # Define your collection names here:
            _self.collections = {
                'production': 'production_2021_2024_hourly',
                'consumption': 'consumption_2022_2024_hourly'
            }

    @st.cache_resource
    def _init_connection(_self) -> pymongo.MongoClient | None:
        """Creates a cached MongoDB client connection."""
        try:
            # Check that the MongoDB URI is defined in st.secrets
            if 'MONGO' not in st.secrets or 'uri' not in st.secrets['MONGO']:
                st.error("MongoDB URI is missing in `st.secrets`.")
                return None
            
            client = pymongo.MongoClient(st.secrets['MONGO']['uri'])
            # Validate the connection by pinging the database
            client.admin.command('ping')
            logging.info("MongoDB connection established.")
            return client
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {e}")
            logging.error(f"MongoDB Connection Error: {e}")
            return None
    



# Load and process data
@st.cache_data
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
        
        # 🛠️ RETTELSE HER: Bruk det korrekte samlingsnavnet
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

    return df

# Cache function for loading Open-Meteo data from the API
@st.cache_data(ttl=86400, show_spinner="Loading Open-Meteo data from API...")
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
