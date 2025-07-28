import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import folium_static

# Set page config
st.set_page_config(page_title="Fire Type Classification", layout="wide")

# Title and description
st.title("Classification of Fire Types in India Using MODIS Satellite Data (2021–2023)")
st.markdown("""
This application predicts the type of fire in India based on MODIS satellite data. 
The underlying model is a Random Forest Classifier trained on data from 2021 to 2023.
""")

# How to use in Google Colab
st.sidebar.header("How to use in Google Colab")
st.sidebar.info("""
1.  **Upload your data:** In your Colab notebook, upload the `modis_2021_India.csv`, `modis_2022_India.csv`, and `modis_2023_India.csv` files.
2.  **Copy and paste the code:** Copy the code from your Jupyter notebook and paste it into cells in your Colab notebook.
""")


# Function to load and preprocess data
@st.cache_data
def load_data():
    # Load datasets
    try:
        df1 = pd.read_csv('modis_2021_India.csv')
        df2 = pd.read_csv('modis_2022_India.csv')
        df3 = pd.read_csv('modis_2023_India.csv')
        df = pd.concat([df1, df2, df3], ignore_index=True)
    except FileNotFoundError:
        st.error("Make sure the CSV files ('modis_2021_India.csv', 'modis_2022_India.csv', 'modis_2023_India.csv') are in the same directory as the app.")
        return None, None

    # Preprocessing
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['month'] = df['acq_date'].dt.month
    df['day'] = df['acq_date'].dt.day
    df['year'] = df['acq_date'].dt.year

    # Label Encoding for satellite and daynight
    le = LabelEncoder()
    df['satellite'] = le.fit_transform(df['satellite'])
    df['daynight'] = le.fit_transform(df['daynight'])
    
    # Drop unnecessary columns
    df = df.drop(columns=['instrument', 'version', 'acq_date'])

    # Define features and target
    X = df.drop('type', axis=1)
    y = df['type']
    
    return df, X, y

# Load data
df, X, y = load_data()

if df is not None:
    # Train the model
    @st.cache_resource
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model(X, y)

    # Sidebar for user input
    st.sidebar.header("Input Fire Characteristics")
    latitude = st.sidebar.slider("Latitude", float(df['latitude'].min()), float(df['latitude'].max()), float(df['latitude'].mean()))
    longitude = st.sidebar.slider("Longitude", float(df['longitude'].min()), float(df['longitude'].max()), float(df['longitude'].mean()))
    brightness = st.sidebar.slider("Brightness (K)", float(df['brightness'].min()), float(df['brightness'].max()), float(df['brightness'].mean()))
    scan = st.sidebar.slider("Scan", float(df['scan'].min()), float(df['scan'].max()), float(df['scan'].mean()))
    track = st.sidebar.slider("Track", float(df['track'].min()), float(df['track'].max()), float(df['track'].mean()))
    acq_time = st.sidebar.slider("Acquisition Time (UTC)", int(df['acq_time'].min()), int(df['acq_time'].max()), int(df['acq_time'].mean()))
    satellite = st.sidebar.selectbox("Satellite", ["Terra", "Aqua"])
    confidence = st.sidebar.slider("Confidence (%)", int(df['confidence'].min()), int(df['confidence'].max()), int(df['confidence'].mean()))
    bright_t31 = st.sidebar.slider("Brightness Temperature Channel 31 (K)", float(df['bright_t31'].min()), float(df['bright_t31'].max()), float(df['bright_t31'].mean()))
    frp = st.sidebar.slider("Fire Radiative Power (MW)", float(df['frp'].min()), float(df['frp'].max()), float(df['frp'].mean()))
    daynight = st.sidebar.selectbox("Day/Night", ["Day", "Night"])
    month = st.sidebar.slider("Month", 1, 12, 6)
    day = st.sidebar.slider("Day", 1, 31, 15)
    year = st.sidebar.slider("Year", 2021, 2023, 2022)

    # Prediction
    if st.sidebar.button("Predict Fire Type"):
        # Encode categorical inputs
        satellite_encoded = 1 if satellite == "Terra" else 0
        daynight_encoded = 0 if daynight == "Day" else 1

        # Create input dataframe
        input_data = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'brightness': [brightness],
            'scan': [scan],
            'track': [track],
            'acq_time': [acq_time],
            'satellite': [satellite_encoded],
            'confidence': [confidence],
            'bright_t31': [bright_t31],
            'frp': [frp],
            'daynight': [daynight_encoded],
            'month': [month],
            'day': [day],
            'year': [year]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        fire_type_mapping = {0: 'Vegetation Fire', 1: 'Volcano', 2: 'Static Land Source', 3: 'Offshore'}
        predicted_fire_type = fire_type_mapping.get(prediction, "Unknown")
        
        st.subheader("Prediction")
        st.write(f"The predicted fire type is: **{predicted_fire_type}**")

    # Display map of fire locations
    st.subheader("Map of Fire Incidents (2021-2023)")
    
    # Sample data for better performance
    df_sample = df.sample(n=5000, random_state=42)

    # Create map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # Add points to map
    for idx, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)

    # Display map
    folium_static(m)

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.write(df)

