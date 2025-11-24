import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

st.set_page_config(page_title="Energy Production Analysis", layout="wide")

st.title("Energy Production Analysis")

# MongoDB connection
@st.cache_resource
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

# Load data
df = load_data()

# Get unique values for filters
price_areas = sorted(df['priceArea'].unique())
production_groups = sorted(df['productionGroup'].unique())
months = sorted(df['month'].unique())
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# Create two columns
col1, col2 = st.columns(2)

# Pie Chart
with col1:
    st.subheader("Production Distribution by Type")
    
    # Radio buttons for price area selection
    selected_area = st.radio(
        "Select Price Area:",
        options=price_areas,
        horizontal=True
    )
    
    # Calculate total production by group for selected area
    area_data = df[df['priceArea'] == selected_area]
    production_summary = area_data.groupby('productionGroup')['quantityKwh'].sum().reset_index()
    production_summary.columns = ['productionGroup', 'total_production']
    production_summary = production_summary.sort_values('total_production', ascending=False)
    
    total = production_summary['total_production'].sum()
    production_summary['percentage'] = production_summary['total_production'] / total * 100
    
    # Define small contributors threshold (<5%)
    small_threshold = 5  # percentage
    small_groups = production_summary[production_summary['percentage'] < small_threshold]
    large_groups = production_summary[production_summary['percentage'] >= small_threshold]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # --- Pie chart 1: All production groups ---
    fig_all = go.Figure(data=[go.Pie(
        labels=[f"{row['productionGroup']} ({row['percentage']:.1f}%)" for _, row in production_summary.iterrows()],
        values=production_summary['total_production'],
        hole=0,
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Production: %{value:,.0f} kWh<extra></extra>',
        marker=dict(colors=colors, line=dict(color='white', width=2))
    )])
    fig_all.update_layout(
        title=f"Total Production Distribution in {selected_area}",
        title_x=0.5,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True
    )
    st.plotly_chart(fig_all, use_container_width=True)
    
    # --- Pie chart 2: Smaller contributors only ---
    if not small_groups.empty:
        fig_small = go.Figure(data=[go.Pie(
            labels=[f"{row['productionGroup']} ({row['percentage']:.1f}%)" for _, row in small_groups.iterrows()],
            values=small_groups['total_production'],
            hole=0,
            textinfo='none',
            hovertemplate='<b>%{label}</b><br>Production: %{value:,.0f} kWh<extra></extra>',
            marker=dict(colors=colors[:len(small_groups)], line=dict(color='white', width=2))
        )])
        fig_small.update_layout(
            title=f"Smaller Production Groups in {selected_area} (<5% of Total)",
            title_x=0.5,
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=True
        )
        st.plotly_chart(fig_small, use_container_width=True)
    else:
        st.info("No smaller contributors (<5%) for this price area.")
    
    # Display summary metrics
    st.metric("Total Production (kWh)", f"{production_summary['total_production'].sum():,.0f}")

# Line Plot
with col2:
    st.subheader("Production Over Time")

    # Pills for production group selection
    selected_groups = st.pills(
        "Select Production Groups:",
        options=production_groups,
        selection_mode="multi",
        default=production_groups[:5]  # Default to first 5 groups
    )

    # Ensure at least one group is selected
    if not selected_groups:
        selected_groups = [production_groups[0]]

    # Month selection
    selected_month = st.selectbox(
        "Select Month:",
        options=list(range(1, 13)),  # 1-12
        format_func=lambda x: month_names[x-1],
        index=0
    )

    # Filter data
    filtered_df = df[
        (df['priceArea'] == selected_area) &
        (df['productionGroup'].isin(selected_groups)) &
        (df['month'] == selected_month)
    ]

    if not filtered_df.empty:
        # Pivot data for line plotting
        pivot_df = filtered_df.pivot(index='startTime', columns='productionGroup', values='quantityKwh')
        
        # Determine small contributors (<5% of total)
        total_per_group = pivot_df.sum()
        threshold = total_per_group.sum() * 0.05
        small_groups = total_per_group[total_per_group < threshold].index.tolist()
        
        # --- Plot 1: All selected groups ---
        fig_all = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, group in enumerate(selected_groups):
            if group in pivot_df.columns:
                group_data = pivot_df[group].sort_index()
                fig_all.add_trace(go.Scatter(
                    x=group_data.index,
                    y=group_data.values,
                    mode='lines+markers',
                    name=group,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
        fig_all.update_layout(
            title=f"Energy Production in {selected_area} - {month_names[selected_month-1]} (All Groups)",
            xaxis_title="Time",
            yaxis_title="Quantity (kWh)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # --- Plot 2: Smaller contributors only ---
        if small_groups:
            fig_small = go.Figure()
            for i, group in enumerate(small_groups):
                group_data = pivot_df[group].sort_index()
                fig_small.add_trace(go.Scatter(
                    x=group_data.index,
                    y=group_data.values,
                    mode='lines+markers',
                    name=group,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
            fig_small.update_layout(
                title=f"Energy Production in {selected_area} - {month_names[selected_month-1]} (Smaller Groups)",
                xaxis_title="Time",
                yaxis_title="Quantity (kWh)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_small, use_container_width=True)
        else:
            st.info("No smaller contributors below threshold to display.")
    else:
        st.warning("No data available for the selected filters.")
        