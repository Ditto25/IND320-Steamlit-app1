import streamlit as st
import pandas as pd
import numpy as np
from StreamlitApplication.Data_loader import load_data

# Page configuration (must be before other st.* display calls)
st.set_page_config(
    page_title="Data Table - Weather Data",
    layout="wide"
)


def pretty_name(col: str) -> str:
    """Convert a column name to a nicer display name (remove underscores, title case)."""
    return col.replace("_", " ").title()


def get_weather_data():
    """Get weather data from session state if available, otherwise fall back to loader."""
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        return st.session_state.weather_data
    # Fallback to existing loader for backwards compatibility
    try:
        return load_data()
    except Exception:
        return None


def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'time' column of datetime64[ns].
    Automatically converts known timestamp-like columns.
    """
    df = df.copy()

    # Case 1: If 'time' already exists
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        return df

    # Case 2: Common timestamp column names
    for candidate in ['timestamp', 'datetime', 'date', 'period_start', 'valid_time']:
        if candidate in df.columns:
            df['time'] = pd.to_datetime(df[candidate], errors='coerce', utc=True)
            return df

    # Case 3: If index is datetime-like
    if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        return df

    # Case 4: No usable time info found
    st.warning("⚠️ No 'time' or timestamp-like column found in DataFrame.")
    return df



def get_first_month_data(df: pd.DataFrame, column: str, max_points: int = 31 * 24):
    """
    Extract the first calendar month of data for a specific column.
    Returns up to max_points rows (default up to 31 days * 24 hours).
    """
    df = ensure_time_column(df)
    if 'time' not in df.columns or df.empty:
        return pd.DataFrame({'time': [], column: []})

    # --- Force proper datetime conversion and remove timezone ---
    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['time'])

    # --- Get first timestamp's month and year ---
    first_ts = df['time'].iloc[0]
    first_month_mask = (
        (df['time'].dt.month == first_ts.month) &
        (df['time'].dt.year == first_ts.year)
    )

    first_month = df.loc[first_month_mask, ['time', column]].head(max_points)
    return first_month


# Main page content
st.title("Weather Data Table with First Month Trends")
st.markdown("---")

data = get_weather_data()

if data is None or len(data) == 0:
    st.warning("⚠️ No weather data loaded. Please visit the page that downloads weather data first.")
    st.info("Once you download data on that page, it will be available here for viewing.")
else:
    st.success(f"✅ Weather data loaded: {len(data)} records")
    if 'selected_area' in st.session_state:
        sel_city = st.session_state.get('selected_city', '')
        st.info(f"📍 Data for: **{st.session_state.selected_area}** ({sel_city})")

try:
    # Work on a copy and ensure time column if possible
    df = data.copy()
    df = ensure_time_column(df)
    

    # Select numeric columns (exclude 'time' if present)
    data_columns = [col for col in df.columns if col != 'time' and pd.api.types.is_numeric_dtype(df[col])]
    # --- Build the summary table first ---
    table_data = []
    # iterate numeric columns previously selected
    for column in data_columns:
        # convert to numeric, coercing errors to NaN, and skip if no valid values
        s = pd.to_numeric(df[column], errors='coerce')
        if s.dropna().empty:
            continue

        col_mean = s.mean()
        col_std = s.std()
        col_min = s.min()
        col_max = s.max()

        first_month_df = get_first_month_data(df, column)

        if first_month_df.empty:
            st.warning(f"No valid data for first month trend of column '{column}'.")
            y_values = []
        else:
            y_values = first_month_df[column].dropna().astype(float).tolist()
            if not y_values:
                st.warning(f"No valid numeric data for first month trend of column '{column}'.")
                continue

            # If we have valid y_values, we can proceed
            pass    
        if len(y_values) == 0:
            st.warning(f"No valid numeric data for first month trend of column '{column}'.")
            continue


        table_data.append({
            "Variable Name": pretty_name(column),
            "Mean": f"{col_mean:.2f}" if pd.notna(col_mean) else "",
            "Std Dev": f"{col_std:.2f}" if pd.notna(col_std) else "",
            "Min": f"{col_min:.2f}" if pd.notna(col_min) else "",
            "Max": f"{col_max:.2f}" if pd.notna(col_max) else "",
            "First Month Trend": y_values
        })

    # Convert to dataframe
    display_df = pd.DataFrame(table_data)

    # --- Display one single dataframe (outside any loop) ---
    st.header("Dataset with First Month Line Charts")
    st.caption("Each row represents a variable from the dataset, with a line chart showing the first month of data.")

    st.dataframe(
        display_df,
        hide_index=True,
        width='stretch',
        column_config={
            "Variable Name": st.column_config.TextColumn("Variable Name", width="medium"),
            "Mean": st.column_config.TextColumn("Mean", width="small"),
            "Std Dev": st.column_config.TextColumn("Std Dev", width="small"),
            "Min": st.column_config.TextColumn("Min", width="small"),
            "Max": st.column_config.TextColumn("Max", width="small"),
            "First Month Trend": st.column_config.LineChartColumn(
                "First Month Trend",
                width="large",
                help="Visualization of the first month of data"
            ),
        }
    )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
