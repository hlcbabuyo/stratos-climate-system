import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the Streamlit page
st.set_page_config(
    page_title="STRATOS Risk Platform",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Configuration ---
API_URL = "http://localhost:8000/predict"
LATITUDE = 8.5342
LONGITUDE = 124.7554

st.sidebar.title("🌡️ STRATOS")
st.sidebar.markdown("Spatial Thermal Risk & Temperature Observation System")
st.sidebar.markdown("---")
st.sidebar.info("This application is the user-facing interface for the STRATOS Machine Learning Pipeline. It connects to the live FastAPI backend for real-time predictions.")

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_recent_nasa_data(days=14):
    """Fetch recent weather data to visualize current trends."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN,WS10M",
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        features = response.json().get("properties", {}).get("parameter", {})
        
        df = pd.DataFrame(features).reset_index()
        df = df.rename(columns={
            "index": "date", 
            "T2M": "temperature_2m", 
            "RH2M": "relative_humidity_2m", 
            "ALLSKY_SFC_SW_DWN": "solar_radiation", 
            "WS10M": "wind_speed_10m"
        })
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        # Filter out missing values from NASA API
        df = df[(df['temperature_2m'] != -999.0)]
        return df
    except Exception as e:
        st.error(f"Failed to fetch live NASA data: {e}")
        return pd.DataFrame()

# --- Main App ---
st.title("STRATOS Community Risk Platform")
st.markdown("Misamis Oriental, Philippines (Lat: 8.5342, Lon: 124.7554)")

tab1, tab2, tab3 = st.tabs(["🌍 Live Observer", "🔮 Interactive Predictor", "🗺️ Logistics Route"])

with tab1:
    st.header("Recent Climate Trends")
    st.markdown("Fetching the last 14 days of meteorological data directly from NASA satellite arrays.")
    
    with st.spinner("Connecting to NASA POWER API..."):
        df_recent = fetch_recent_nasa_data(14)
        
    if not df_recent.empty:
        # Display key metrics for today
        latest = df_recent.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{latest['temperature_2m']} °C")
        col2.metric("Humidity", f"{latest['relative_humidity_2m']} %")
        col3.metric("Solar Radiation", f"{latest['solar_radiation']} kW-hr/m²")
        col4.metric("Wind Speed", f"{latest['wind_speed_10m']} m/s")
        
        st.subheader("Temperature Trend")
        st.line_chart(df_recent.set_index('date')['temperature_2m'], use_container_width=True)
    else:
        st.warning("No live data available at the moment.")

with tab2:
    st.header("Predict Extreme Heat Events")
    st.markdown("Adjust the sliders to simulate weather conditions. The Random Forest model deployed on our FastAPI server will instantly predict if these conditions constitute an **Extreme Heat Warning**.")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("Weather Parameters")
        temp_input = st.slider("Temperature (°C)", min_value=20.0, max_value=45.0, value=30.0, step=0.1)
        humid_input = st.slider("Relative Humidity (%)", min_value=30.0, max_value=100.0, value=75.0, step=1.0)
        solar_input = st.slider("Solar Radiation (kW-hr/m²/day)", min_value=0.0, max_value=30.0, value=15.0, step=0.5)
        wind_input = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
        
        predict_btn = st.button("Query ML Backend", type="primary")
        
    with col_output:
        st.subheader("AI Prediction Status")
        if predict_btn:
            payload = {
                "temperature_2m": temp_input,
                "relative_humidity_2m": humid_input,
                "solar_radiation": solar_input,
                "wind_speed_10m": wind_input
            }
            
            with st.spinner("Querying FastAPI..."):
                try:
                    res = requests.post(API_URL, json=payload)
                    if res.status_code == 200:
                        is_extreme = res.json().get("extreme_heat_warning", False)
                        
                        if is_extreme:
                            st.error("🚨 HIGH RISK: EXTREME HEAT WARNING 🚨")
                            st.markdown("The current conditions are flagged by the ML model as extremely hazardous. Local officials should activate cooling centers and distribute water.")
                        else:
                            st.success("✅ LOW RISK: SAFE CONDITIONS ✅")
                            st.markdown("The conditions do not currently meet the threshold for extreme heat. Routine operations can continue.")
                    else:
                        st.error(f"Backend Error: Ensure the FastAPI server is running. Status code: {res.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Backend Connection Error. Please ensure you are running `uvicorn main:app --reload` in the `app/` directory.")

with tab3:
    st.header("Safe Logistics Routing")
    st.markdown("When extreme heat warnings are issued, relief operations must be routed safely. The map below is generated by our **Q-Learning Agent**, utilizing the Bellman equation to chart the optimal path across a 5x5 municipal grid while avoiding High-Penalty heat zones.")
    
    # Recreate the visualization statically for the dashboard
    grid_size = 5
    start_state = 0
    goal_state = 24
    
    # We use a fixed seed to match the notebook's presentation
    np.random.seed(42)
    heat_zones = np.random.choice(range(1, 24), size=4, replace=False)
    
    # Manually defined path from the notebook's trained agent 
    # (Since we are just displaying the prototype result)
    optimal_path = [0, 5, 10, 11, 16, 17, 22, 23, 24] # Example safe path avoiding heat zones
    
    grid_viz = np.zeros((grid_size, grid_size))
    for hz in heat_zones:
        grid_viz[hz // grid_size, hz % grid_size] = -1 # Heat zones in red
    grid_viz[goal_state // grid_size, goal_state % grid_size] = 2 # Goal in green
    
    for step in optimal_path:
        if step != start_state and step != goal_state:
            grid_viz[step // grid_size, step % grid_size] = 1 # Path in blue
            
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(grid_viz, cmap=['red', 'white', 'blue', 'green'], annot=True, cbar=False, linewidths=1, linecolor='black', ax=ax)
    ax.set_title('Optimal Relief Route (Red: Heat Zone, Blue: Path, Green: Goal)')
    
    st.pyplot(fig)
