from pathlib import Path
import pandas as pd
import streamlit as st

DATAFILE = Path(__file__).with_name("open-meteo-subset.csv") #It links to the CSV file in the same directory as this script
#could have also used: DATAFILE = Path("DataApp/open-meteo-subset.csv")

@st.cache_data(show_spinner=False) # Cache the loaded data to optimize performance 
def load_data() -> pd.DataFrame:
    """Load local CSV, parse time, set as index, and return a tidy DataFrame."""  
    # Check if the data file exists
    if not DATAFILE.exists():
        st.error(f"The data file was not found: {DATAFILE}")
        st.stop()  # Stop the app if the file is missing

    df = pd.read_csv(DATAFILE, parse_dates=["time"]) # Read the CSV file and parse the "time" column as datetime
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df