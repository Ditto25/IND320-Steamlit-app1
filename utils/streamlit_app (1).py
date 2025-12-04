
import streamlit as st

# Set global page configuration
st.set_page_config(
    page_title="Energy Analysis Platform", 
    page_icon="⚡",
    layout="wide"
)

# --- 1. Define all Page objects (st.Page) ---

# Core Analysis Group
elhub_page = st.Page("pages/elhub_data.py", title="Elhub Raw Data", icon="📊")
stl_page = st.Page("pages/stl_spec.py", title="STL & Spectral Analysis", icon="📉")
table_page = st.Page("pages/data_table.py", title="Data Table View", icon="📑")
production_page = st.Page("pages/production_analysis.py", title="Production Analysis", icon="🏭")
outliers_page = st.Page("pages/outliers.py", title="Outlier Management", icon="🧹")
weather_page = st.Page("pages/weather_energy.py", title="Weather Correlation", icon="🌦️")

# Geospatial Analysis (Your Map Group)
map_page = st.Page("pages/map_snowdrift.py", title="Map and Snowdrift", icon="❄️")

# Energy Analytics (Your SARIMAX Group)
forecast_page = st.Page("pages/sarimax_forecast.py", title="SARIMAX Forecasting", icon="🔮")


# --- 2. Set up navigation with Grouped Sections (st.navigation) ---

# The keys in this dictionary become the section titles in the sidebar.
pg = st.navigation({
    # Group 1: Core Data and Initial Analysis
    "Data Analysis": [
        elhub_page, 
        table_page, 
        stl_page,
        production_page,
        outliers_page,
        weather_page,
    ],
    # Group 2: Geospatial
    "Geospatial Analysis": [
        map_page
    ],
    # Group 3: Forecasting
    "Energy Analytics": [
        forecast_page
    ]
})

# --- 3. Run the App ---
# This generates the sidebar and manages page routing
pg.run()