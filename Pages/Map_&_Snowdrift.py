from __future__ import annotations
from copy import deepcopy
from datetime import date, timedelta
from typing import Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import folium
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

# --- Imports from your utils ---
from utils.Data_loader import (
    load_pricearea_geojson,
    load_elhub_production_data,
    load_elhub_consumption_data,
    # Sørg for at denne funksjonen laster ned data for et gitt ÅR
    load_open_meteo_api, 
)
from utils.Snow_drift import (
    compute_yearly_results,
    compute_average_sector,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CONSTANTS
# -----------------------------------------------------------------------------
# Dette er Page-modulen: Map_Snow_Drift_Analysis.py
# (Jeg antar at koden du sendte er main-funksjonen til denne filen)
st.set_page_config(page_title="Map & Snow Drift Analysis", layout="wide")

PRICEAREA_GEO_KEY = "ElSpotOmr"
VALID_PRICEAREAS = {"NO1", "NO2", "NO3", "NO4", "NO5"}

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (MAP & DATA)
# ... (mean_by_pricearea, build_map er uendret) ...
# -----------------------------------------------------------------------------
def get_groups(kind: str) -> list[str]:
    """Return sorted list of groups for the given kind."""
    if kind == "production":
        df = load_elhub_production_data()
    else:
        df = load_elhub_consumption_data()
    
    field = "group"
    if df.empty or field not in df.columns:
        return []

    groups = sorted(df[field].dropna().unique())
    return groups

def mean_by_pricearea(
    kind: str,
    group: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Compute mean quantity per price area."""
    if kind == "production":
        df = load_elhub_production_data()
    else:
        df = load_elhub_consumption_data()

    group_field = "group"
    
    if df.empty:
        return pd.DataFrame(columns=["priceArea", "mean_kwh"])

    mask = (
        (df[group_field] == group)
        & (df["startTime"] >= start_ts)
        & (df["startTime"] < end_ts)
    )
    
    df_sel = df.loc[mask, ["priceArea", "quantityKwh"]]

    if df_sel.empty:
        return pd.DataFrame(columns=["priceArea", "mean_kwh"])

    agg = (
        df_sel.groupby("priceArea", as_index=False)["quantityKwh"]
        .mean()
        .rename(columns={"quantityKwh": "mean_kwh"})
    )
    agg['mean_gwh'] = agg['mean_kwh'] / 1_000_000
    # Endrer navnet tilbake til 'pricearea' for å matche build_map
    agg = agg.rename(columns={"priceArea": "pricearea"})
    
    return agg.drop(columns=['mean_kwh'])

def build_map(
    geojson: dict,
    df_mean: pd.DataFrame,
    selected_pricearea: str,
    clicked_coord: dict | None,
) -> folium.Map:
    """Build Folium map locked to Norway region."""
    gj = deepcopy(geojson)

    value_by_area: dict[str, float] = {}
    if not df_mean.empty:
        value_by_area = df_mean.set_index("pricearea")["mean_gwh"].to_dict()

    for feat in gj.get("features", []):
        props = feat.setdefault("properties", {})
        pa_raw = props.get(PRICEAREA_GEO_KEY, "")
        pa_clean = pa_raw.replace(" ", "") if isinstance(pa_raw, str) else pa_raw
        props["pricearea_clean"] = pa_clean
        props["mean_gwh"] = float(value_by_area.get(pa_clean, float("nan")))

    
    m = folium.Map(
        location=[60.0, 10.0],          # Sentrert over Norge
        zoom_start=4, 
        min_zoom=3,                     # Hindrer å zoome for langt inn
        max_zoom=12,                    # Hindrer å zoome for langt ut
        max_bounds=False,               # Låser panorering til bounds
        min_lat=57, max_lat=72,         # Setter harde grenser
        min_lon=2, max_lon=33,
        tiles="cartodbpositron"
    )

    if not df_mean.empty:
        df_choro = df_mean.rename(columns={"pricearea": "pricearea_clean"})
        folium.Choropleth(
            geo_data=gj,
            data=df_choro,
            columns=["pricearea_clean", "mean_gwh"],
            key_on="feature.properties.pricearea_clean",
            fill_color="YlOrRd",
            fill_opacity=0.6,
            nan_fill_opacity=0.0,
            line_opacity=0.0,
            legend_name="Mean quantity (GWh)",
        ).add_to(m)

    def style_function(feature):
        pa_clean = feature["properties"].get("pricearea_clean")
        if pa_clean == selected_pricearea:
            return {"fillOpacity": 0.0, "color": "red", "weight": 3}
        else:
            return {"fillOpacity": 0.0, "color": "black", "weight": 1}

    folium.GeoJson(
        gj,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=["pricearea_clean", "mean_gwh"],
            aliases=["Price area", "Mean quantity (GWh)"],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)

    if clicked_coord is not None:
        folium.Marker(
            location=[clicked_coord["lat"], clicked_coord["lon"]],
            popup=f"Lat: {clicked_coord['lat']:.4f}, Lon: {clicked_coord['lon']:.4f}",
            tooltip="Click to analyze snow drift here",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    return m

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS (SNOW DRIFT)
# -----------------------------------------------------------------------------

@st.cache_data
def download_weather_for_seasons(
    latitude: float,
    longitude: float,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    
    # 1. Bestem det siste gyldige kalenderåret
    current_year = date.today().year
    
    # Vi kan kun laste ned data for kalenderår til og med inneværende år.
    last_valid_year = current_year 

    # 2. Definer hvilke år vi skal hente basert på sesong og begrensninger
    # Sesonger: fra start_season til end_season (f.eks. 2022 til 2024)
  
    # Vi må filtrere bort år som er større enn last_valid_year
    years: List[int] = []
    for year in range(start_season, end_season + 1):
        if year <= last_valid_year:
            years.append(year)
        else:
            # Ignorerer fremtidige år
            st.warning(f"Skipper år {year} da det er i fremtiden.")


    frames: List[pd.DataFrame] = []

    for year in years:
        try:
            # load_open_meteo_api skal hente et fullt år (start 1. jan, slutt 31. des)
            df_year = load_open_meteo_api(latitude=latitude, longitude=longitude, year=year)
            frames.append(df_year)
        except Exception as e:
            # Hvis API-et feiler for et år, for eksempel fordi det er for nylig og ufullstendig,
            # logger vi advarselen og går videre.
            st.warning(f"Kunne ikke laste data for år {year}: {e}") 

    # ... (resten av funksjonen er den samme) ...
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    return df_all
def prepare_snowdrift_dataframe(
    df_weather: pd.DataFrame,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    """Prepare DataFrame for Snow_drift.py functions."""
    if df_weather.empty:
        return pd.DataFrame()

    df = df_weather.copy().reset_index()    # index -> 'time'
    df.rename(
        columns={
            "time": "time",
            "temperature_2m": "temperature_2m (°C)",
            "precipitation": "precipitation (mm)",
            "wind_speed_10m": "wind_speed_10m (m/s)",
            "wind_direction_10m": "wind_direction_10m (°)",
        },
        inplace=True,
    )

    # Season logic: July–Dec = current year, Jan–June = previous year
    df["season"] = df["time"].apply(
        lambda dt: dt.year if dt.month >= 7 else dt.year - 1
    )

    # Filtrerer kun for de valgte sesongene
    mask = (df["season"] >= start_season) & (df["season"] <= end_season)
    df = df.loc[mask].reset_index(drop=True)
    return df

def plot_yearly_snow_transport(yearly_df: pd.DataFrame):
    df_plot = yearly_df.copy()
    df_plot["Qt (tonnes/m)"] = df_plot["Qt (kg/m)"] / 1000.0

    fig = px.bar(
        df_plot,
        x="season",
        y="Qt (tonnes/m)",
        labels={"season": "Season", "Qt (tonnes/m)": "Qt (tonnes/m)"},
        title="Yearly mean snow transport per season",
    )
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=40, b=40))
    fig.update_xaxes(tickangle=-45)
    return fig

def plot_wind_rose(avg_sector_values: np.ndarray, overall_avg_kgm: float):
    num_sectors = len(avg_sector_values)
    if num_sectors == 0:
        return go.Figure()

    values_tonnes = np.array(avg_sector_values) / 1000.0
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", 
                      "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    theta_deg = np.linspace(0, 360, num_sectors, endpoint=False)

    fig = go.Figure(
        data=go.Barpolar(
            r=values_tonnes,
            theta=theta_deg,
            text=directions,
            hovertemplate="Direction: %{text}<br>Qt: %{r:.3f} tonnes/m<extra></extra>",
        )
    )

    overall_tonnes = overall_avg_kgm / 1000.0
    fig.update_layout(
        title=f"Avg directional snow transport<br>Overall Qt = {overall_tonnes:,.1f} tonnes/m",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        polar=dict(
            angularaxis=dict(
                tickmode="array", tickvals=theta_deg, ticktext=directions,
                direction="clockwise", rotation=90
            ),
            radialaxis=dict(angle=90, tickangle=90, showline=True, linewidth=1),
        ),
        showlegend=False,
    )
    return fig

# -----------------------------------------------------------------------------
# 4. MAIN APPLICATION
# -----------------------------------------------------------------------------

def main():
    st.title("Price Areas Map & Snow Drift Analysis")
    
    # 1.1 Load Data
    try:
        geojson = load_pricearea_geojson()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        
    # 1.2 Sidebar/Top Controls for Map
    with st.expander("Map Settings (Data Selection)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            data_type_label = st.radio("Data type", ["Production", "Consumption"], horizontal=True, key="map_data_type")
            kind = "production" if data_type_label == "Production" else "consumption"
            
            # --- DYNAMISK DATO-LOGIKK ---
            if kind == "production":
                min_date_limit = date(2022, 1, 1)
                min_season_limit = 2022
            else:
                min_date_limit = date(2021, 1, 1)
                min_season_limit = 2021
            
            groups = get_groups(kind)
            if not groups:
                st.warning("No groups found.")
                group = None
            else:
                group = st.selectbox("Group", groups, key="map_group_select")

            selected_pricearea = st.session_state.get("pricearea", "NO2")
            if selected_pricearea not in VALID_PRICEAREAS:
                selected_pricearea = "NO2"

        with col2:
            # 🚨 FIX 1: Setter max dato til I DAG
            MAX_DATE_LIMIT = date.today()
            
            start_date = st.date_input(
                "Start date", 
                value=max(date(2023, 1, 1), min_date_limit), 
                min_value=min_date_limit, 
                max_value=MAX_DATE_LIMIT,
                key="map_start_date"
            )
            days = st.slider("Interval length (days)", 1, 365, 30, key="map_interval")

    # 1.3 Compute Map Data
    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + timedelta(days=days)
    
    df_mean = pd.DataFrame()
    if group:
        df_mean = mean_by_pricearea(kind=kind, group=group, start_ts=start_ts, end_ts=end_ts)
    

    # 1.4 Render Map
    st.subheader("Map")
    st.caption("Click anywhere on the map to select a location for snow drift analysis.")
    
    clicked_coord = st.session_state.get("map_coord")
    
    folium_map = build_map(
        geojson=geojson,
        df_mean=df_mean,
        selected_pricearea=selected_pricearea,
        clicked_coord=clicked_coord,
    )

    # Display Map
    map_state = st_folium(folium_map, width=1200, height=600)

    # 1.5 Handle Click Events
    if map_state and map_state.get("last_clicked"):
        click = map_state["last_clicked"]
        new_coord = {"lat": click["lat"], "lon": click["lng"]}
        
        if new_coord != clicked_coord:
            st.session_state["map_coord"] = new_coord
            st.rerun()

    # --- PART 2: SNOW DRIFT ANALYSIS ---
    st.divider()
    st.header("Snow Drift Analysis")

    # 2.1 Check for Coordinate
    coord = st.session_state.get("map_coord")
    
    if not coord:
        st.info("👋 Please click on the map above to select a location for snow drift analysis.")
        st.stop()

    lat, lon = coord["lat"], coord["lon"]
    st.success(f"📍 Analyzing location: Lat {lat:.4f}, Lon {lon:.4f}")

    # 2.2 Snow Drift Controls
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.subheader("Settings")
        
        # 🚨 FIX 2: Setter MAX_SEASON til inneværende år
        MAX_SEASON = date.today().year-1
        default_start = max(min_season_limit, 2022) 
        
        start_season, end_season = st.slider(
            "Season range", 
            min_value=min_season_limit, 
            max_value=MAX_SEASON, 
            # Endrer default-verdien til å inkludere 2024 hvis mulig
            value=(default_start, max(2023, MAX_SEASON)), 
            key="snow_season_slider"
        )
        
        with st.expander("Model Parameters", expanded=False):
            T = st.number_input("Max transport distance T (m)", 500.0, 10000.0, 3000.0, 100.0, key="snow_T")
            F = st.number_input("Fetch distance F (m)", 1000.0, 100000.0, 30000.0, 1000.0, key="snow_F")
            theta = st.slider("Relocation coefficient θ", 0.0, 1.0, 0.5, 0.05, key="snow_theta")
        
        compute_btn = st.button("Compute Snow Drift", type="primary", use_container_width=True, key="snow_compute_btn")

    with col_s2:
        if compute_btn:
            with st.spinner(f"Downloading weather data for {lat:.2f}, {lon:.2f}..."):
                try:
                    # Bruker den justerte download_weather_for_seasons som tolererer feil
                    df_weather = download_weather_for_seasons(lat, lon, start_season, end_season)
                except Exception as exc:
                    st.error(f"Error downloading data: {exc}")
                    st.stop()

            if df_weather.empty:
                st.error("No weather data found.")
                st.stop()
            
            df_snow = prepare_snowdrift_dataframe(df_weather, start_season, end_season)
            
            if df_snow.empty:
                st.warning("No data in selected season range.")
                st.stop()
                
            yearly_df = compute_yearly_results(df_snow, T=T, F=F, theta=theta)
            
            if not yearly_df.empty:
                # 2.3 Display Results
                tab1, tab2, tab3 = st.tabs(["Charts", "Wind Rose", "Data"])
                
                with tab1:
                    fig_yearly = plot_yearly_snow_transport(yearly_df)
                    st.plotly_chart(fig_yearly, use_container_width=True)
                
                with tab2:
                    avg_sectors = compute_average_sector(df_snow)
                    overall_avg = float(yearly_df["Qt (kg/m)"].mean())
                    fig_rose = plot_wind_rose(avg_sectors, overall_avg)
                    st.plotly_chart(fig_rose, use_container_width=True)
                    
                with tab3:
                    display_df = yearly_df[["season", "Qt (kg/m)", "Control"]].copy()
                    display_df["Qt (tonnes/m)"] = display_df["Qt (kg/m)"] / 1000
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("Calculation returned no results.")
        else:
            st.info("Adjust settings and click 'Compute Snow Drift' to see results.")

if __name__ == "__main__":
    main()