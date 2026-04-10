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
st.set_page_config(page_title="Neurological Stroke Detection", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Hemiparesis & Stroke Detector")
st.markdown("Live 3D Biomechanical Tracking, Kinematics, & Explainable AI Diagnostics.")

# ---------------------------------------------------------
# 2. LOAD THE AI MODEL AND SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_ai_assets():
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
    st.sidebar.warning("⚠️ Demo Mode (AI Model files not detected)")

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
# 4. ADVANCED 3D BIOMECHANICS & PATIENT PROFILES
# ---------------------------------------------------------
def generate_stickman_frame(t, profile, noise_level):
    walk_cycle = t * np.pi * 2
    head, neck, pelvis = [0, 0, 1.8], [0, 0, 1.5], [0, 0, 0.9]
    r_shoulder, l_shoulder = [0.3, 0, 1.5], [-0.3, 0, 1.5]
    r_hip, l_hip = [0.2, 0, 0.9], [-0.2, 0, 0.9]
    
    if profile == "Healthy Control":
        r_elbow = [0.4, -0.3 * np.sin(walk_cycle), 1.2]
        r_wrist = [0.4, -0.6 * np.sin(walk_cycle), 0.9]
        r_knee = [0.2, 0.3 * np.sin(walk_cycle + np.pi/2), 0.5]
        r_ankle = [0.2, 0.4 * np.sin(walk_cycle + np.pi/2), 0.1]
        l_elbow = [-0.4, 0.3 * np.sin(walk_cycle), 1.2]
        l_wrist = [-0.4, 0.6 * np.sin(walk_cycle), 0.9]
        l_knee = [-0.2, -0.3 * np.sin(walk_cycle + np.pi/2), 0.5]
        l_ankle = [-0.2, -0.4 * np.sin(walk_cycle + np.pi/2), 0.1]
        
    elif profile == "Severe Left Hemiparesis":
        r_elbow = [0.4, -0.3 * np.sin(walk_cycle), 1.2]
        r_wrist = [0.4, -0.6 * np.sin(walk_cycle), 0.9]
        r_knee = [0.2, 0.3 * np.sin(walk_cycle + np.pi/2), 0.5]
        r_ankle = [0.2, 0.4 * np.sin(walk_cycle + np.pi/2), 0.1]
        noise = np.random.normal(0, noise_level, 3)
        l_elbow = [-0.35, 0.0 + noise[0], 1.3] 
        l_wrist = [-0.2, 0.1 + noise[1], 1.4]  
        l_knee = [-0.2, -0.1 * np.sin(walk_cycle + np.pi/2), 0.5] 
        l_ankle = [-0.2, -0.1 * np.sin(walk_cycle + np.pi/2), 0.05] 

    elif profile == "Mild Right Spasticity":
        l_elbow = [-0.4, 0.3 * np.sin(walk_cycle), 1.2]
        l_wrist = [-0.4, 0.6 * np.sin(walk_cycle), 0.9]
        l_knee = [-0.2, -0.3 * np.sin(walk_cycle + np.pi/2), 0.5]
        l_ankle = [-0.2, -0.4 * np.sin(walk_cycle + np.pi/2), 0.1]
        noise = np.random.normal(0, noise_level * 2, 3)
        r_elbow = [0.4, -0.1 * np.sin(walk_cycle) + noise[0], 1.2]
        r_wrist = [0.4, -0.2 * np.sin(walk_cycle) + noise[1], 1.0]
        r_knee = [0.2, 0.2 * np.sin(walk_cycle + np.pi/2), 0.5]
        r_ankle = [0.2, 0.3 * np.sin(walk_cycle + np.pi/2), 0.1]

    elif profile == "Bilateral Bradykinesia (Parkinsonian)":
        noise = np.random.normal(0, noise_level * 1.5, 3)
        r_elbow = [0.4, -0.05 * np.sin(walk_cycle) + noise[0], 1.2]
        r_wrist = [0.4, -0.1 * np.sin(walk_cycle) + noise[1], 0.9]
        r_knee = [0.2, 0.1 * np.sin(walk_cycle + np.pi/2), 0.4]
        r_ankle = [0.2, 0.1 * np.sin(walk_cycle + np.pi/2), 0.1]
        l_elbow = [-0.4, 0.05 * np.sin(walk_cycle) + noise[0], 1.2]
        l_wrist = [-0.4, 0.1 * np.sin(walk_cycle) + noise[1], 0.9]
        l_knee = [-0.2, -0.1 * np.sin(walk_cycle + np.pi/2), 0.4]
        l_ankle = [-0.2, -0.1 * np.sin(walk_cycle + np.pi/2), 0.1]

    x = [head[0], neck[0], r_shoulder[0], r_elbow[0], r_wrist[0], None, neck[0], l_shoulder[0], l_elbow[0], l_wrist[0], None, neck[0], pelvis[0], r_hip[0], r_knee[0], r_ankle[0], None, pelvis[0], l_hip[0], l_knee[0], l_ankle[0]]
    y = [head[1], neck[1], r_shoulder[1], r_elbow[1], r_wrist[1], None, neck[1], l_shoulder[1], l_elbow[1], l_wrist[1], None, neck[1], pelvis[1], r_hip[1], r_knee[1], r_ankle[1], None, pelvis[1], l_hip[1], l_knee[1], l_ankle[1]]
    z = [head[2], neck[2], r_shoulder[2], r_elbow[2], r_wrist[2], None, neck[2], l_shoulder[2], l_elbow[2], l_wrist[2], None, neck[2], pelvis[2], r_hip[2], r_knee[2], r_ankle[2], None, pelvis[2], l_hip[2], l_knee[2], l_ankle[2]]
    
    return x, y, z, l_wrist, r_wrist

def build_animated_stickman(frames_data, color_scheme):
    fig = go.Figure(
        data=[go.Scatter3d(x=frames_data[0][0], y=frames_data[0][1], z=frames_data[0][2], mode='lines+markers', line=dict(color=color_scheme, width=6))],
        layout=go.Layout(
            scene=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[0, 2]), aspectmode='cube'),
            updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play Kinematics", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, transition=dict(duration=0))])])],
            margin=dict(l=0, r=0, b=0, t=0), height=450
        ),
        frames=[go.Frame(data=[go.Scatter3d(x=f[0], y=f[1], z=f[2])]) for f in frames_data]
    )
    return fig

# ---------------------------------------------------------
# 5. DASHBOARD CONTROLS (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("📡 Live Telemetry Control")
patient_profiles = ("Healthy Control", "Severe Left Hemiparesis", "Mild Right Spasticity", "Bilateral Bradykinesia (Parkinsonian)")
patient_type = st.sidebar.radio("Select Patient Profile:", patient_profiles)

st.sidebar.divider()
st.sidebar.subheader("Simulation Parameters")
num_frames = st.sidebar.slider("Sampling Resolution (Frames)", min_value=40, max_value=120, value=80, step=10)
noise_level = st.sidebar.slider("Neurological Tremor Intensity", min_value=0.0, max_value=0.15, value=0.03, step=0.01)

if st.sidebar.button("▶️ Initialize AI Biomechanical Scan"):
    with st.spinner(f"Rendering {num_frames} High-Fidelity Kinematic Frames..."):
        
        time_steps = np.linspace(0, 2, num_frames)
        healthy_frames = [generate_stickman_frame(t, "Healthy Control", 0) for t in time_steps]
        patient_frames = [generate_stickman_frame(t, patient_type, noise_level) for t in time_steps]
        
        raw_data = np.array([np.concatenate([f[3], f[4]]) for f in patient_frames])
        features_12d = calculate_clinical_features(raw_data)
        
        dt = time_steps[1] - time_steps[0]
        left_wrist_y_vel = np.gradient(raw_data[:, 1], dt)
        left_wrist_y_acc = np.gradient(left_wrist_y_vel, dt)
        right_wrist_y_vel = np.gradient(raw_data[:, 4], dt)
        right_wrist_y_acc = np.gradient(right_wrist_y_vel, dt)
        
        indices = np.linspace(0, num_frames - 1, 20, dtype=int)
        scan_window = features_12d[indices]
        
        prediction_prob = 0.0
        is_stroke = False
        
        if model and scaler:
            scaled_features = scaler.transform(scan_window).reshape(1, 20, 12)
            prediction_prob = model.predict(scaled_features)[0][0]
            is_stroke = prediction_prob > 0.5
        else:
            if patient_type == "Healthy Control":
                prediction_prob = 0.04
            elif patient_type == "Severe Left Hemiparesis":
                prediction_prob = 0.98
                is_stroke = True
            elif patient_type == "Mild Right Spasticity":
                prediction_prob = 0.76
                is_stroke = True
            else:
                prediction_prob = 0.62
                is_stroke = True

        # ---------------------------------------------------------
        # 6. UI TABS: CLINICAL, XAI, AND ARCHITECTURE
        # ---------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["🩺 Clinical View (3D)", "⚙️ Live AI Diagnostics", "📚 The Science (Architecture)"])
        
        # --- TAB 1: CLINICAL VIEW ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Neural Network Confidence", f"{prediction_prob * 100:.2f}%")
            if is_stroke:
                col2.error(f"🚨 ALERT: Abnormal Gait Detected ({patient_type})")
            else:
                col2.success("✅ STATUS: Healthy Symmetric Gait")
            col3.metric("Peak Lateral Balance Shift", f"{np.max(features_12d[:, 11]):.3f}")

            st.subheader("Live 3D Spatial Tracking (60 FPS)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Baseline (Healthy Reference)**")
                st.plotly_chart(build_animated_stickman(healthy_frames, '#00CC96'), use_container_width=True) 
            with c2:
                st.markdown(f"**Live Patient ({patient_type})**")
                patient_color = '#EF553B' if is_stroke else '#00CC96' 
                st.plotly_chart(build_animated_stickman(patient_frames, patient_color), use_container_width=True)

        # --- TAB 2: EXPLAINABLE AI ---
        with tab2:
            st.markdown("### Deep Learning & Biomechanical Breakdown")
            
            st.subheader("1. CNN Tremor Detection: High-Frequency Acceleration (m/s²)")
            st.markdown("The CNN acts as an anomaly detector, scanning for erratic, high-frequency jitters embedded inside the slower walking wave.")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=left_wrist_y_acc, mode='lines', name='Left Arm Acceleration', line=dict(color='red' if is_stroke and "Left" in patient_type else 'cyan', width=2)))
            fig_acc.add_trace(go.Scatter(y=right_wrist_y_acc, mode='lines', name='Right Arm Acceleration', line=dict(color='orange' if is_stroke and "Right" in patient_type else 'lightgreen', width=2)))
            fig_acc.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_acc, use_container_width=True)

            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.subheader("2. CNN Spasticity Filter (Amplitude)")
                st.write("Detects instances where one arm's wave is 'flattened' or compressed due to muscle stiffness.")
                if model:
                    fig_heat = px.imshow(scaled_features[0].T, color_continuous_scale="inferno", aspect="auto")
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    fake_heat = np.random.rand(12, 20) if not is_stroke else np.random.rand(12, 20) * 3
                    fig_heat = px.imshow(fake_heat, color_continuous_scale="inferno", aspect="auto")
                    st.plotly_chart(fig_heat, use_container_width=True)
                
            with col_x2:
                st.subheader("3. LSTM Bradykinesia Filter (Rhythm)")
                st.write("Evaluates temporal rhythm, checking if kinetic energy drops or is unnaturally sluggish.")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(y=features_12d[:, 9], mode='lines', name='Left Arm Energy', line=dict(color='red' if is_stroke and "Left" in patient_type else 'cyan', width=3)))
                fig_lstm.add_trace(go.Scatter(y=features_12d[:, 10], mode='lines', name='Right Arm Energy', line=dict(color='orange' if is_stroke and "Right" in patient_type else 'lightgreen', width=3, dash='dot')))
                fig_lstm.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_lstm, use_container_width=True)

        # --- TAB 3: THE SCIENCE (FROM YOUR SCREENSHOTS) ---
        with tab3:
            st.header("🧠 AI Architecture & Clinical Methodology")
            st.markdown("The AI analyzes **240 data points** per prediction (12 features × 20 timesteps).")
            
            with st.expander("🔬 The 12 Clinical Biomarkers", expanded=True):
                st.markdown("""
                Before the AI makes a guess, the `calculate_clinical_features` function translates raw numbers into 12 specific indicators:
                * **Raw Kinematics (Channels 0-5):** The raw X, Y, and Z acceleration of both the left and right arms.
                * **Positional Drift (Channels 6-8):** The mathematical difference between the arms (`left_xyz - right_xyz`), tracking if one arm is drifting or dragging.
                * **Kinetic Energy / Arm Weakness (Channels 9-10):** The pure magnitude of force generated by each arm, directly powering the Arm Weakness Monitor.
                * **Lateral Balance Shift (Channel 11):** The Asymmetry Index ratio, indicating if the patient's center of gravity is dangerously skewed to one side.
                """)

            with st.expander("🧬 AI Architecture: Hybrid CNN-LSTM", expanded=True):
                st.markdown("""
                Once the 12 features are calculated, the Hybrid CNN-LSTM network scans them across 1.5 seconds of continuous motion, hunting for three clinical stroke signatures:
                * **Bradykinesia (Sluggishness):** The LSTM (Long Short-Term Memory) layer evaluates the rhythm of the wave. The AI checks if movement has suddenly become unnaturally sluggish or delayed on one side.
                * **Spasticity (Muscle Stiffness):** The CNN (Convolutional Neural Network) checks the amplitude (height) of the wave. The AI looks for instances where one arm's wave is "flattened" or compressed, indicating the arm is stiff and not swinging fully.
                * **Acute Tremors (Involuntary Shaking):** The CNN acts as a high-frequency anomaly detector, scanning for erratic, high-frequency jitters embedded inside the broader, slower walking wave.
                """)
                
            st.info("💡 **Detecting Clinical Signatures:** When one arm presents with lower kinetic magnitude, disrupted temporal rhythm, high-frequency micro-shakes, and a flattened swing compared to the healthy arm, the model outputs a probability close to 1.0 and triggers the CRITICAL ANOMALY alert.")

else:
    st.info("👈 Select a Patient Profile and click **Initialize AI Biomechanical Scan**.")
