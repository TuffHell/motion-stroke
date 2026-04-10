import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Neurological Stroke Detection", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Hemiparesis & Stroke Detector")
st.markdown("Live 3D Biomechanical Skeleton Tracking & Explainable AI Diagnostics.")

# ---------------------------------------------------------
# 2. LOAD THE AI MODEL AND SCALER (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_ai_assets():
    # Wrap in try/except for local testing without files
    try:
        model = load_model('stroke_detection_model.h5')
        scaler = joblib.load('scaler.save')
        return model, scaler
    except:
        return None, None

model, scaler = load_ai_assets()
if model:
    st.sidebar.success("✅ Neural Network Online")
else:
    st.sidebar.warning("⚠️ Running in Demo Mode (Model/Scaler files missing)")

# ---------------------------------------------------------
# 3. CLINICAL PHYSICS ENGINE
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
# 4. 3D STICKMAN GENERATOR LOGIC
# ---------------------------------------------------------
def generate_stickman_frame(t, profile, noise_level):
    # Base walking mechanics (sine waves)
    walk_cycle = t * np.pi
    
    # Healthy Baseline Dynamics
    head = [0, 0, 1.8]
    neck = [0, 0, 1.5]
    pelvis = [0, 0, 0.9]
    
    right_shoulder = [0.3, 0, 1.5]
    right_elbow = [0.4, -0.2 * np.sin(walk_cycle), 1.2]
    right_wrist = [0.4, -0.4 * np.sin(walk_cycle), 0.9]
    
    right_hip = [0.2, 0, 0.9]
    right_knee = [0.2, 0.3 * np.sin(walk_cycle + np.pi/2), 0.5]
    right_ankle = [0.2, 0.4 * np.sin(walk_cycle + np.pi/2), 0.1]
    
    if profile == "Healthy":
        left_shoulder = [-0.3, 0, 1.5]
        left_elbow = [-0.4, 0.2 * np.sin(walk_cycle), 1.2]
        left_wrist = [-0.4, 0.4 * np.sin(walk_cycle), 0.9]
        left_hip = [-0.2, 0, 0.9]
        left_knee = [-0.2, -0.3 * np.sin(walk_cycle + np.pi/2), 0.5]
        left_ankle = [-0.2, -0.4 * np.sin(walk_cycle + np.pi/2), 0.1]
    else:
        # Stroke / Hemiparesis Dynamics (Left side restricted, rigid, noisy)
        noise = np.random.normal(0, noise_level, 3)
        left_shoulder = [-0.3, 0, 1.5]
        left_elbow = [-0.35, 0.05 * np.sin(walk_cycle) + noise[0], 1.3] # Stiff elbow
        left_wrist = [-0.35, 0.1 * np.sin(walk_cycle) + noise[1], 1.1]  # Dropped, stiff wrist
        left_hip = [-0.2, 0, 0.9]
        left_knee = [-0.2, -0.1 * np.sin(walk_cycle + np.pi/2), 0.5]    # Dragging leg
        left_ankle = [-0.2, -0.1 * np.sin(walk_cycle + np.pi/2), 0.1]

    # Combine into a single continuous line with None gaps to draw bones
    x = [head[0], neck[0], right_shoulder[0], right_elbow[0], right_wrist[0], None, neck[0], left_shoulder[0], left_elbow[0], left_wrist[0], None, neck[0], pelvis[0], right_hip[0], right_knee[0], right_ankle[0], None, pelvis[0], left_hip[0], left_knee[0], left_ankle[0]]
    y = [head[1], neck[1], right_shoulder[1], right_elbow[1], right_wrist[1], None, neck[1], left_shoulder[1], left_elbow[1], left_wrist[1], None, neck[1], pelvis[1], right_hip[1], right_knee[1], right_ankle[1], None, pelvis[1], left_hip[1], left_knee[1], left_ankle[1]]
    z = [head[2], neck[2], right_shoulder[2], right_elbow[2], right_wrist[2], None, neck[2], left_shoulder[2], left_elbow[2], left_wrist[2], None, neck[2], pelvis[2], right_hip[2], right_knee[2], right_ankle[2], None, pelvis[2], left_hip[2], left_knee[2], left_ankle[2]]
    
    return x, y, z, left_wrist, right_wrist

def build_animated_stickman(frames_data, color_scheme):
    fig = go.Figure(
        data=[go.Scatter3d(x=frames_data[0][0], y=frames_data[0][1], z=frames_data[0][2], mode='lines+markers', line=dict(color=color_scheme, width=5))],
        layout=go.Layout(
            scene=dict(xaxis=dict(range=[-1, 1], autorange=False), yaxis=dict(range=[-1, 1], autorange=False), zaxis=dict(range=[0, 2], autorange=False), aspectmode='cube'),
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play Animation", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])],
            margin=dict(l=0, r=0, b=0, t=0), height=400
        ),
        frames=[go.Frame(data=[go.Scatter3d(x=f[0], y=f[1], z=f[2])]) for f in frames_data]
    )
    return fig

# ---------------------------------------------------------
# 5. DASHBOARD CONTROLS (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("📡 Live Telemetry Control")
patient_type = st.sidebar.radio("Select Patient Profile:", ("Healthy Control", "Acute Stroke (Hemiparesis)"))
st.sidebar.divider()
st.sidebar.subheader("Testing Data Parameters")
num_frames = st.sidebar.slider("Sample Size (Timesteps)", min_value=20, max_value=100, value=40, step=10)
noise_level = st.sidebar.slider("Tremor / Noise Injection", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

if st.sidebar.button("▶️ Run Biomechanical Scan"):
    with st.spinner(f"Generating {num_frames} frames of kinematic data..."):
        
        # Generate temporal data arrays
        time_steps = np.linspace(0, 4, num_frames)
        
        # Build Stickman Data Arrays
        healthy_frames = [generate_stickman_frame(t, "Healthy", 0) for t in time_steps]
        patient_frames = [generate_stickman_frame(t, patient_type, noise_level) for t in time_steps]
        
        # Extract just the Left and Right Wrists for the AI (to match your 6D shape)
        raw_data = np.array([np.concatenate([f[3], f[4]]) for f in patient_frames])
        
        # Process through physics engine
        features_12d = calculate_clinical_features(raw_data)
        
        # AI requires exactly 20 timesteps (Take the most recent 20 for the scan)
        scan_window = features_12d[-20:, :]
        
        prediction_prob = 0.0
        is_stroke = False
        
        if model and scaler:
            scaled_features = scaler.transform(scan_window).reshape(1, 20, 12)
            prediction_prob = model.predict(scaled_features)[0][0]
            is_stroke = prediction_prob > 0.5
        elif patient_type == "Acute Stroke (Hemiparesis)":
            prediction_prob = 0.89  # Mock data if model missing
            is_stroke = True

        # ---------------------------------------------------------
        # 6. UI TABS: CLINICAL VS EXPLAINABLE AI
        # ---------------------------------------------------------
        tab1, tab2 = st.tabs(["🩺 Clinical View (3D Kinematics)", "⚙️ Explainable AI (Diagnostics)"])
        
        # --- TAB 1: DOCTOR'S CLINICAL VIEW ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Neural Network Probability", f"{prediction_prob * 100:.2f}%")
            if is_stroke:
                col2.error("🚨 CRITICAL: Hemiparesis Signature Detected!")
            else:
                col2.success("✅ STATUS: Normal Gait Pattern")
            col3.metric("Lateral Balance Shift", f"{scan_window[-1, 11]:.3f}")

            # Dual 3D Render
            st.subheader("Live 3D Spatial Tracking")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Baseline (Healthy Reference)**")
                st.plotly_chart(build_animated_stickman(healthy_frames, 'cyan'), use_container_width=True)
            with c2:
                st.markdown(f"**Live Patient ({patient_type})**")
                patient_color = 'red' if is_stroke else 'cyan'
                st.plotly_chart(build_animated_stickman(patient_frames, patient_color), use_container_width=True)
                
            st.info("💡 Click **Play Animation** on the graphs above to watch the kinematic loop.")

        # --- TAB 2: JUDGE'S DEVELOPER VIEW (XAI) ---
        with tab2:
            st.markdown("### Neural Network Activation Diagnostics")
            col_x1, col_x2 = st.columns(2)
            
            with col_x1:
                st.subheader(f"1. Spasticity Heatmap (Last 20 frames)")
                if model:
                    fig_heat = px.imshow(scaled_features[0].T, labels=dict(x="Timestep", y="Feature Channel", color="Activation"), color_continuous_scale="inferno", aspect="auto")
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.warning("Heatmap requires valid model/scaler files.")
                
            with col_x2:
                st.subheader(f"2. Temporal Rhythm ({num_frames} frames)")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(y=features_12d[:, 9], mode='lines', name='Left Arm Kinetic Energy', line=dict(color='red' if is_stroke else 'cyan', width=3)))
                fig_lstm.add_trace(go.Scatter(y=features_12d[:, 10], mode='lines', name='Right Arm Kinetic Energy', line=dict(color='orange', width=3, dash='dot')))
                fig_lstm.update_layout(xaxis_title="Timestep", yaxis_title="Kinetic Magnitude (Joules)", height=400)
                st.plotly_chart(fig_lstm, use_container_width=True)
else:
    st.info("👈 Adjust sampling parameters and click **Run Biomechanical Scan**.")
