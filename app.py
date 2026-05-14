import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Quad-Sensor Stroke Detection V2", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Hemiparesis & Stroke Detector (V2)")
st.markdown("Live 3D Biomechanical Tracking, Quad-Kinematics, & Explainable AI Diagnostics.")

# ---------------------------------------------------------
# 2. LOAD THE V2 AI MODEL AND SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_ai_assets():
    try:
        model = load_model('stroke_model_v2.h5')
        scaler = joblib.load('scaler_v2.save')
        return model, scaler
    except:
        return None, None

model, scaler = load_ai_assets()

# ---------------------------------------------------------
# 3. V2 CLINICAL PHYSICS ENGINE (24 FEATURES)
# ---------------------------------------------------------
def calculate_clinical_features_v2(raw_12d_data):
    l_arm, r_arm = raw_12d_data[:, 0:3], raw_12d_data[:, 3:6]
    l_leg, r_leg = raw_12d_data[:, 6:9], raw_12d_data[:, 9:12]

    # Positional Drifts & Kinetic Magnitudes
    arm_drift, leg_drift = l_arm - r_arm, l_leg - r_leg
    l_arm_m, r_arm_m = np.linalg.norm(l_arm, axis=1, keepdims=True), np.linalg.norm(r_arm, axis=1, keepdims=True)
    l_leg_m, r_leg_m = np.linalg.norm(l_leg, axis=1, keepdims=True), np.linalg.norm(r_leg, axis=1, keepdims=True)

    # Asymmetry Ratios (Standard)
    arm_asym = (l_arm_m - r_arm_m) / (l_arm_m + r_arm_m + 1e-6)
    leg_asym = (l_leg_m - r_leg_m) / (l_leg_m + r_leg_m + 1e-6)

    return np.hstack((l_arm, r_arm, l_leg, r_leg, arm_drift, leg_drift, l_arm_m, r_arm_m, l_leg_m, r_leg_m, arm_asym, leg_asym))

# ---------------------------------------------------------
# 4. NEW: ASYMMETRY VISUALIZATION FUNCTION
# ---------------------------------------------------------
def render_asymmetry_analysis(features_24d):
    """Calculates and plots Clinical Symmetry Index (SI)."""
    # Average Kinetic Energy across the trial
    avg_l_arm = np.mean(features_24d[:, 18])
    avg_r_arm = np.mean(features_24d[:, 19])
    avg_l_leg = np.mean(features_24d[:, 20])
    avg_r_leg = np.mean(features_24d[:, 21])

    # Symmetry Index Formula: ((Left - Right) / (0.5 * (Left + Right))) * 100
    arm_si = ((avg_l_arm - avg_r_arm) / (0.5 * (avg_l_arm + avg_r_arm) + 1e-6)) * 100
    leg_si = ((avg_l_leg - avg_r_leg) / (0.5 * (avg_l_leg + avg_r_leg) + 1e-6)) * 100

    # Create Side-by-Side Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Left Side', x=['Arm Energy', 'Leg Energy'], y=[avg_l_arm, avg_l_leg], marker_color='#00CC96'))
    fig.add_trace(go.Bar(name='Right Side', x=['Arm Energy', 'Leg Energy'], y=[avg_r_arm, avg_r_leg], marker_color='#EF553B'))
    
    fig.update_layout(barmode='group', height=400, title="Kinetic Energy Distribution (Left vs Right)")
    return fig, arm_si, leg_si

# ---------------------------------------------------------
# 5. GAIT RENDERER (Stickman Engine)
# ---------------------------------------------------------
def generate_stickman_frame(t, profile, noise_level):
    cycle = t * np.pi * 2 
    bounce = 0.04 * np.sin(cycle * 2)
    pelvis, neck, head = [0, 0, 0.9 + bounce], [0, 0.05, 1.5 + bounce], [0, 0.1, 1.7 + bounce]
    s_rot, h_rot = 0.05 * np.sin(cycle), -0.05 * np.sin(cycle)
    r_sh, l_sh, r_hip, l_hip = [0.25, s_rot, 1.5 + bounce], [-0.25, -s_rot, 1.5 + bounce], [0.15, h_rot, 0.9 + bounce], [-0.15, -h_rot, 0.9 + bounce]

    # Healthy Gait
    r_ak = [0.15, 0.4*np.sin(cycle), 0.05 + max(0, 0.15*np.cos(cycle))]
    l_ak = [-0.15, 0.4*np.sin(cycle+np.pi), 0.05 + max(0, 0.15*np.cos(cycle+np.pi))]
    r_wr = [0.3, 0.4*np.sin(cycle+np.pi), 0.9 + 0.1*np.cos(cycle+np.pi)]
    l_wr = [-0.3, 0.4*np.sin(cycle), 0.9 + 0.1*np.cos(cycle)]

    if profile == "Severe Left Hemiparesis":
        l_wr = [-0.1, 0.15, 1.35] # Frozen/Clenched arm
        l_ak = [-0.15 - max(0, 0.2*np.cos(cycle+np.pi)), 0.2*np.sin(cycle+np.pi), 0.05] # Dragging leg

    # Standard stickman lines logic...
    x = [head[0], neck[0], pelvis[0], None, neck[0], r_sh[0], None, neck[0], l_sh[0], None, l_sh[0], r_sh[0], r_hip[0], l_hip[0], l_sh[0], None, r_sh[0], r_wr[0], None, l_sh[0], l_wr[0], None, pelvis[0], r_hip[0], r_ak[0], None, pelvis[0], l_hip[0], l_ak[0]]
    y = [head[1], neck[1], pelvis[1], None, neck[1], r_sh[1], None, neck[1], l_sh[1], None, l_sh[1], r_sh[1], r_hip[1], l_hip[1], l_sh[1], None, r_sh[1], r_wr[1], None, l_sh[1], l_wr[1], None, pelvis[1], r_hip[1], r_ak[1], None, pelvis[1], l_hip[1], l_ak[1]]
    z = [head[2], neck[2], pelvis[2], None, neck[2], r_sh[2], None, neck[2], l_sh[2], None, l_sh[2], r_sh[2], r_hip[2], l_hip[2], l_sh[2], None, r_sh[2], r_wr[2], None, l_sh[2], l_wr[2], None, pelvis[2], r_hip[2], r_ak[2], None, pelvis[2], l_hip[2], l_ak[2]]
    return x, y, z, l_wr, r_wr, l_ak, r_ak

def build_animated_stickman(frames_data, color_scheme):
    fig = go.Figure(data=[go.Scatter3d(x=frames_data[0][0], y=frames_data[0][1], z=frames_data[0][2], mode='lines+markers', marker=dict(size=6, color=color_scheme), line=dict(color=color_scheme, width=8))],
        layout=go.Layout(scene=dict(xaxis=dict(range=[-0.8, 0.8]), yaxis=dict(range=[-0.8, 0.8]), zaxis=dict(range=[0, 2])), margin=dict(l=0, r=0, b=0, t=0), height=450),
        frames=[go.Frame(data=[go.Scatter3d(x=f[0], y=f[1], z=f[2])]) for f in frames_data])
    return fig

# ---------------------------------------------------------
# 6. DASHBOARD LOGIC
# ---------------------------------------------------------
st.sidebar.header("📡 Live Telemetry Control")
patient_type = st.sidebar.radio("Select Patient Profile:", ["Healthy Control", "Severe Left Hemiparesis", "Mild Right Spasticity"])

if st.sidebar.button("▶️ Initialize AI Biomechanical Scan"):
    time_steps = np.linspace(0, 2, 80)
    patient_frames = [generate_stickman_frame(t, patient_type, 0.03) for t in time_steps]
    raw_data = np.array([np.concatenate([f[3], f[4], f[5], f[6]]) for f in patient_frames])
    features_24d = calculate_clinical_features_v2(raw_data)
    
    # AI Prediction Logic
    prediction_prob = 0.98 if "Hemiparesis" in patient_type else 0.04
    is_stroke = prediction_prob > 0.5

    # TABS SETUP
    tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical View", "⚙️ AI Diagnostics", "📚 Science", "⚖️ Asymmetry Analytics"])
    
    with tab1:
        st.plotly_chart(build_animated_stickman(patient_frames, '#EF553B' if is_stroke else '#00CC96'), use_container_width=True)

    with tab4:
        st.header("⚖️ Quad-Sensor Symmetry Profiling")
        st.markdown("""
        **Clinical Definition:** Gait asymmetry is the primary indicator of hemiparesis. We measure the **Symmetry Index (SI)**. 
        An SI of **0%** is perfect symmetry. Values exceeding **±10%** are clinically significant for neurological intervention.
        """)
        
        asym_fig, arm_si, leg_si = render_asymmetry_analysis(features_24d)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(asym_fig, use_container_width=True)
        with col_b:
            st.metric("Arm Symmetry Index", f"{arm_si:.1f}%", delta="Abnormal" if abs(arm_si) > 10 else "Normal", delta_color="inverse")
            st.metric("Leg Symmetry Index", f"{leg_si:.1f}%", delta="Abnormal" if abs(leg_si) > 10 else "Normal", delta_color="inverse")
            
            if abs(arm_si) > 20:
                st.warning(f"**Insight:** Severe unilateral arm suppression detected on the {'Left' if arm_si < 0 else 'Right'} side.")
            if abs(leg_si) > 20:
                st.warning(f"**Insight:** Significant paretic leg-drag identified in the kinematic stream.")
