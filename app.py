import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import time
import os

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="InnoHealth Quad-Sensor V2", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Quad-Sensor Kinematic Telemetry Dashboard (V2)")
st.write("Full-body neurological tracking utilizing Hybrid CNN-LSTM to scan for Clinical Stroke Signatures.")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_ai_assets():
    try:
        model = load_model('stroke_model_v2.h5')
        scaler = joblib.load('scaler_v2.save')
        return model, scaler, True
    except:
        return None, None, False

dashboard_model, loaded_scaler, assets_loaded = load_ai_assets()

# --- 3. KINEMATIC SIMULATION & SIGNATURE ENGINE ---
def simulate_v2_data(scenario):
    """Simulates 240 data points (24 features x 20 timesteps) based on clinical signatures."""
    t = np.linspace(0, 2*np.pi, 20)
    window = np.zeros((20, 12)) # 12 raw channels
    
    # Base gait
    window[:, 0] = np.sin(t) * 0.5  # L Arm X
    window[:, 3] = np.sin(t + np.pi) * 0.5  # R Arm X
    window[:, 6] = np.sin(t + np.pi) * 0.6  # L Leg X
    window[:, 9] = np.sin(t) * 0.6  # R Leg X
    
    signatures = {"Bradykinesia": "Normal", "Spasticity": "None", "Tremors": "None"}
    
    if scenario == "Pathology (Stroke Simulation)":
        # Apply Sluggishness (Bradykinesia) - temporal rhythm disruption
        window[:, 3] = np.sin(t * 0.5 + np.pi) * 0.2 
        # Apply Muscle Stiffness (Spasticity) - flattened swing
        window[:, 9] *= 0.15 
        # Apply Tremors - high frequency micro-shakes
        window[:, 3] += np.random.normal(0, 0.05, 20) 
        
        signatures = {"Bradykinesia": "⚠️ DETECTED", "Spasticity": "⚠️ DETECTED", "Tremors": "⚠️ DETECTED"}

    # Feature Engineering (creating the 24 features)
    l_arm, r_arm = window[:, 0:3], window[:, 3:6]
    l_leg, r_leg = window[:, 6:9], window[:, 9:12]
    
    # Biomarker Calculations
    drift = l_arm - r_arm # Positional Drift
    l_mag, r_mag = np.linalg.norm(l_arm, axis=1, keepdims=True), np.linalg.norm(r_arm, axis=1, keepdims=True)
    balance_shift = (l_mag - r_mag) / (l_mag + r_mag + 1e-6) # Lateral Balance Shift
    
    # Placeholder for legs to reach 24 features
    leg_drift = l_leg - r_leg
    ll_mag, rl_mag = np.linalg.norm(l_leg, axis=1, keepdims=True), np.linalg.norm(r_leg, axis=1, keepdims=True)
    leg_asym = (ll_mag - rl_mag) / (ll_mag + rl_mag + 1e-6)
    
    full_24 = np.hstack((window, drift, leg_drift, l_mag, r_mag, ll_mag, rl_mag, balance_shift, leg_asym))
    scaled = loaded_scaler.transform(full_24).reshape(1, 20, 24)
    prob = dashboard_model.predict(scaled, verbose=0)[0][0]
    
    return window, signatures, prob, l_mag.mean(), r_mag.mean(), balance_shift.mean()

# --- 4. 3D SKELETON RENDERER ---
def render_3d_skeleton(raw_data):
    # Mapping points
    frame = 10
    l_wrist = [-0.5, raw_data[frame, 0] + 0.8, 1.0]
    r_wrist = [0.5, raw_data[frame, 3] + 0.8, 1.0]
    l_ankle = [-0.2, raw_data[frame, 6] + 0.8, 0.1]
    r_ankle = [0.2, raw_data[frame, 9] + 0.8, 0.1]
    
    # Body Structure
    x = [0, 0, -0.3, -0.4, l_wrist[0], -0.3, 0.3, 0.4, r_wrist[0], 0.3, 0, 0, -0.2, -0.2, l_ankle[0], -0.2, 0.2, 0.2, r_ankle[0]]
    y = [0.8, 0.8, 0.8, 0.8, l_wrist[1], 0.8, 0.8, 0.8, r_wrist[1], 0.8, 0.8, 0.8, 0.8, 0.8, l_ankle[1], 0.8, 0.8, 0.8, r_ankle[1]]
    z = [1.7, 1.5, 1.4, 1.2, 1.0, 1.4, 1.4, 1.2, 1.0, 1.4, 1.5, 0.9, 0.9, 0.5, 0.1, 0.9, 0.9, 0.5, 0.1]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', 
                                       line=dict(color='#e67e22', width=8),
                                       marker=dict(size=6, color='#2ecc71'))])
    fig.update_layout(scene=dict(bgcolor="#111827", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      margin=dict(l=0, r=0, b=0, t=0), height=500, paper_bgcolor="#111827")
    return fig

# --- 5. MAIN UI ---
if not assets_loaded:
    st.error("Assets not found. Please place `stroke_model_v2.h5` and `scaler_v2.save` in this folder.")
    st.stop()

with st.sidebar:
    st.header("Control Center")
    scenario = st.radio("Patient Telemetry Stream:", ["Healthy (Symmetrical Gait)", "Pathology (Stroke Simulation)"])
    run_btn = st.button("Fetch & Analyze Data", type="primary", use_container_width=True)

if run_btn:
    raw, sigs, prob, l_m, r_m, b_shift = simulate_and_predict(scenario) # helper calls simulate_v2_data
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("3D Kinematic Motion Viewer")
        st.plotly_chart(render_3d_skeleton(raw), use_container_width=True)
        
    with col2:
        st.subheader("Clinical Signature Analysis")
        # Signature Detections
        c1, c2, c3 = st.columns(3)
        c1.metric("Bradykinesia", sigs["Bradykinesia"])
        c2.metric("Spasticity", sigs["Spasticity"])
        c3.metric("Tremors", sigs["Tremors"])
        
        st.divider()
        # 12 Biomarker Highlights
        st.subheader("Biomarker Monitoring")
        st.write(f"**Lateral Balance Shift (Index):** {b_shift:.3f}")
        st.progress(abs(b_shift) if abs(b_shift) <= 1 else 1.0)
        
        st.write(f"**Kinetic Magnitude (L vs R Arm):** {l_m:.2f} / {r_m:.2f}")
        st.divider()
        
        # AI Final Result
        if prob > 0.5:
            st.error(f"🚨 CRITICAL ANOMALY: {(prob*100):.1f}% Confidence")
        else:
            st.success(f"✅ NOMINAL STATUS: {(100 - prob*100):.1f}% Confidence")

    # --- 6. EXPORTING CLINICAL RECORD ---
    report_content = f"INNOHEALTH RPM ALERT\nStatus: {'CRITICAL' if prob > 0.5 else 'NORMAL'}\nProb: {prob:.4f}\n"
    st.download_button("📥 Download AI Clinical Record", report_content, file_name="Clinical_Alert.txt")

else:
    st.info("Awaiting telemetry connection. Use the sidebar to fetch patient data.")
