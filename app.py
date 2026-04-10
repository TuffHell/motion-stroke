import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import time

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Neurological Stroke Detection", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Hemiparesis & Stroke Detector")
st.markdown("Real-time biomechanical analysis using a Hybrid CNN-LSTM Neural Network.")

# ---------------------------------------------------------
# 2. LOAD THE AI MODEL AND SCALER (CACHED)
# ---------------------------------------------------------
# @st.cache_resource ensures the AI only loads once when the app starts!
@st.cache_resource
def load_ai_assets():
    model = load_model('stroke_detection_model.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

try:
    model, scaler = load_ai_assets()
    st.sidebar.success("✅ AI Model Loaded Successfully")
except Exception as e:
    st.sidebar.error("❌ Could not load model/scaler. Ensure 'stroke_detection_model.h5' and 'scaler.save' are in the folder.")

# ---------------------------------------------------------
# 3. CLINICAL PHYSICS ENGINE (From your Jupyter code)
# ---------------------------------------------------------
def calculate_clinical_features(raw_6d_data):
    left_xyz = raw_6d_data[:, 0:3]
    right_xyz = raw_6d_data[:, 3:6]
    diffs = left_xyz - right_xyz
    left_mag = np.linalg.norm(left_xyz, axis=1, keepdims=True)
    right_mag = np.linalg.norm(right_xyz, axis=1, keepdims=True)
    asymmetry = (left_mag - right_mag) / (left_mag + right_mag + 1e-6)
    return np.hstack((left_xyz, right_xyz, diffs, left_mag, right_mag, asymmetry))

# ---------------------------------------------------------
# 4. DASHBOARD & SIMULATION LOGIC
# ---------------------------------------------------------
st.sidebar.header("Patient Telemetry Control")

# Button to simulate pulling 1.5 seconds of live data
if st.sidebar.button("📡 Fetch Live Patient Data (1.5s Window)"):
    with st.spinner("Analyzing patient biomechanics..."):
        time.sleep(1) # Fake loading time for dramatic effect
        
        # Simulate 20 timesteps of raw 6D sensor data (Left XYZ, Right XYZ)
        # In reality, this would be replaced by live Bluetooth data or a CSV upload
        simulated_raw_data = np.random.normal(loc=0.0, scale=1.0, size=(20, 6))
        
        # 1. Run through your physics engine
        features_12d = calculate_clinical_features(simulated_raw_data)
        
        # 2. Scale the data using your exported StandardScaler
        # Reshape to 2D for the scaler, then back to 3D (1, 20, 12) for the AI
        scaled_features = scaler.transform(features_12d).reshape(1, 20, 12)
        
        # 3. The AI Prediction!
        prediction_prob = model.predict(scaled_features)[0][0]
        is_stroke = prediction_prob > 0.5

        # ---------------------------------------------------------
        # 5. RENDER RESULTS TO THE DASHBOARD
        # ---------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        
        # Metric 1: The AI Probability
        col1.metric("Neural Network Probability", f"{prediction_prob * 100:.2f}%")
        
        # Metric 2: Clinical Status
        if is_stroke:
            col2.error("CRITICAL: Hemiparesis Signature Detected!")
        else:
            col2.success("STATUS: Normal Gait Pattern")
            
        # Metric 3: Asymmetry Index (from the last timestep)
        current_asymmetry = features_12d[-1, 11]
        col3.metric("Lateral Balance Shift (Asymmetry)", f"{current_asymmetry:.3f}")

        # Draw the Interactive Telemetry Graph (Plotly)
        st.subheader("Live Telemetry: Biomechanical Waveforms")
        fig = go.Figure()
        
        # Plotting Left vs Right Arm Magnitudes (Channels 9 and 10)
        timesteps = list(range(20))
        left_magnitudes = features_12d[:, 9]
        right_magnitudes = features_12d[:, 10]
        
        fig.add_trace(go.Scatter(x=timesteps, y=left_magnitudes, mode='lines+markers', name='Left Arm Kinetic Energy', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=timesteps, y=right_magnitudes, mode='lines+markers', name='Right Arm Kinetic Energy', line=dict(color='orange')))
        
        fig.update_layout(
            xaxis_title="Timesteps (1.5s total)",
            yaxis_title="Kinetic Magnitude",
            plot_bgcolor='rgba(10,10,10,1)',
            paper_bgcolor='rgba(10,10,10,1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Clinical Note:** The Neural Network scans these 20 timesteps across 12 engineered features to detect wave-flattening (spasticity) and rhythm delays (bradykinesia).")
else:
    st.write("👈 Click **Fetch Live Patient Data** in the sidebar to run the AI sequence.")
