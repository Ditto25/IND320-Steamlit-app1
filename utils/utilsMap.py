from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import folium

def load_json(filepath: Path):
    """Load GeoJSON file and extract area codes."""
    with open(filepath, "r", encoding="utf-8") as f:
        geojson_data = gpd.read_file(f)

    return geojson_data

# def display_map(geojson_data):
#     # Ensure WGS84 for folium and pick a safe tooltip field
#     gdf = geojson_data
#     gdf = gdf.to_crs(epsg=4326)
#     tooltip_field = "name" if "name" in gdf.columns else next((c for c in gdf.columns if c != "geometry"), None)

#     minx, miny, maxx, maxy = gdf.total_bounds
#     center_lat = (miny + maxy) / 2.0
#     center_lon = (minx + maxx) / 2.0
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=4.5)

#     folium.GeoJson(
#         data=gdf.__geo_interface__,
#         name="Norwegian Bidding Zones",
#         style_function=lambda x: {"fillColor": "blue", "color": "black", "weight": 1, "fillOpacity": 0.5},
#         tooltip=folium.GeoJsonTooltip(fields=[tooltip_field] if tooltip_field else [], aliases=["Zone"])
#     ).add_to(m)

#     folium.LayerControl().add_to(m)
#     return m


def display_map(
    geojson_data,
    selected_price_area: str | None = None,
    clicked_points: Iterable[Sequence[float]] | None = None,
):
    gdf = geojson_data.to_crs(epsg=4326)
    tooltip_field = "name" if "name" in gdf.columns else next((c for c in gdf.columns if c != "geometry"), None)
    selected_price_area_lc = selected_price_area.lower() if selected_price_area else None

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4.5)

    def style_function(feature):
        props = feature.get("properties", {})
        price_area = str(props.get("Price area") or props.get("price_area") or props.get("Price_area") or "").lower()
        is_selected = selected_price_area_lc and price_area == selected_price_area_lc
        base = {"fillColor": "#2563eb", "color": "#1f2937", "weight": 1, "fillOpacity": 0.45}
        if is_selected:
            base.update({"color": "#f97316", "weight": 3, "fillOpacity": 0.6})
        return base

    folium.GeoJson(
        data=gdf.__geo_interface__,
        name="Norwegian Bidding Zones",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[tooltip_field] if tooltip_field else [], aliases=["Zone"])
    ).add_to(m)

    for point in clicked_points or []:
        if len(point) == 2:
            folium.CircleMarker(location=[point[0], point[1]], radius=4, color="#ef4444", fill=True, fill_opacity=0.9).add_to(m)

    folium.LayerControl().add_to(m)
    return m