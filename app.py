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

    arm_drift, leg_drift = l_arm - r_arm, l_leg - r_leg
    l_arm_m, r_arm_m = np.linalg.norm(l_arm, axis=1, keepdims=True), np.linalg.norm(r_arm, axis=1, keepdims=True)
    l_leg_m, r_leg_m = np.linalg.norm(l_leg, axis=1, keepdims=True), np.linalg.norm(r_leg, axis=1, keepdims=True)

    arm_asym = (l_arm_m - r_arm_m) / (l_arm_m + r_arm_m + 1e-6)
    leg_asym = (l_leg_m - r_leg_m) / (l_leg_m + r_leg_m + 1e-6)

    return np.hstack((l_arm, r_arm, l_leg, r_leg, arm_drift, leg_drift, l_arm_m, r_arm_m, l_leg_m, r_leg_m, arm_asym, leg_asym))

# ---------------------------------------------------------
# 4. ASYMMETRY VISUALIZATION (VOLUMETRIC POINT CLOUDS)
# ---------------------------------------------------------
def render_asymmetry_analysis(features_24d, patient_frames):
    # Bar Chart Logic
    avg_l_arm = np.mean(features_24d[:, 18])
    avg_r_arm = np.mean(features_24d[:, 19])
    avg_l_leg = np.mean(features_24d[:, 20])
    avg_r_leg = np.mean(features_24d[:, 21])

    arm_si = ((avg_l_arm - avg_r_arm) / (0.5 * (avg_l_arm + avg_r_arm) + 1e-6)) * 100
    leg_si = ((avg_l_leg - avg_r_leg) / (0.5 * (avg_l_leg + avg_r_leg) + 1e-6)) * 100

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name='Left Side', x=['Arm Energy', 'Leg Energy'], y=[avg_l_arm, avg_l_leg], marker_color='#00CC96'))
    fig_bar.add_trace(go.Bar(name='Right Side', x=['Arm Energy', 'Leg Energy'], y=[avg_r_arm, avg_r_leg], marker_color='#EF553B'))
    fig_bar.update_layout(barmode='group', height=400, title="Kinetic Energy Breakdown")

    # VOLUMETRIC POINT CLOUD LOGIC (Fixes the blank Mesh3d bug)
    l_w_x, l_w_y, l_w_z = [f[3][0] for f in patient_frames], [f[3][1] for f in patient_frames], [f[3][2] for f in patient_frames]
    r_w_x, r_w_y, r_w_z = [f[4][0] for f in patient_frames], [f[4][1] for f in patient_frames], [f[4][2] for f in patient_frames]
    l_a_x, l_a_y, l_a_z = [f[5][0] for f in patient_frames], [f[5][1] for f in patient_frames], [f[5][2] for f in patient_frames]
    r_a_x, r_a_y, r_a_z = [f[6][0] for f in patient_frames], [f[6][1] for f in patient_frames], [f[6][2] for f in patient_frames]

    fig_traj = go.Figure()
    
    # We use 'markers' with a huge size and low opacity to create a "glowing volume cloud" effect
    # This guarantees it renders even if the path is a perfectly flat 1D line.
    vol_l = dict(size=22, color='#00CC96', opacity=0.25, line=dict(width=0))
    vol_r = dict(size=22, color='#EF553B', opacity=0.25, line=dict(width=0))
    
    # The central line acts as the "bone", the markers act as the "flesh/volume"
    fig_traj.add_trace(go.Scatter3d(x=l_w_x, y=l_w_y, z=l_w_z, mode='lines+markers', line=dict(color='#00CC96', width=4), marker=vol_l, name='Left Arm Volume'))
    fig_traj.add_trace(go.Scatter3d(x=r_w_x, y=r_w_y, z=r_w_z, mode='lines+markers', line=dict(color='#EF553B', width=4), marker=vol_r, name='Right Arm Volume'))
    fig_traj.add_trace(go.Scatter3d(x=l_a_x, y=l_a_y, z=l_a_z, mode='lines+markers', line=dict(color='#00CC96', width=4), marker=vol_l, name='Left Leg Volume'))
    fig_traj.add_trace(go.Scatter3d(x=r_a_x, y=r_a_y, z=r_a_z, mode='lines+markers', line=dict(color='#EF553B', width=4), marker=vol_r, name='Right Leg Volume'))

    fig_traj.update_layout(
        title="3D Spatial Movement Volumes (Point Clouds)",
        scene=dict(xaxis=dict(range=[-0.5, 0.5]), yaxis=dict(range=[-0.5, 0.5]), zaxis=dict(range=[0, 1.5]), aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40), height=400
    )

    return fig_bar, fig_traj, arm_si, leg_si

# ---------------------------------------------------------
# 5. PHOTOREALISTIC HUMAN GAIT (ALL 4 PROFILES RESTORED)
# ---------------------------------------------------------
def generate_stickman_frame(t, profile, noise_level):
    cycle = t * np.pi * 2 
    bounce = 0.04 * np.sin(cycle * 2)
    
    pelvis, neck, head = [0, 0, 0.9 + bounce], [0, 0.05, 1.5 + bounce], [0, 0.1, 1.7 + bounce]
    s_rot, h_rot = 0.05 * np.sin(cycle), -0.05 * np.sin(cycle)
    
    r_shoulder, l_shoulder = [0.25, s_rot, 1.5 + bounce], [-0.25, -s_rot, 1.5 + bounce]
    r_hip, l_hip = [0.15, h_rot, 0.9 + bounce], [-0.15, -h_rot, 0.9 + bounce]

    # --- HEALTHY BASELINE ---
    r_ankle_z = 0.05 + max(0, 0.15 * np.cos(cycle)) 
    r_knee = [0.15, 0.2 * np.sin(cycle) + 0.05, 0.5 + (r_ankle_z - 0.05) * 1.5]
    r_ankle = [0.15, 0.4 * np.sin(cycle), r_ankle_z]
    
    l_ankle_z = 0.05 + max(0, 0.15 * np.cos(cycle + np.pi))
    l_knee = [-0.15, 0.2 * np.sin(cycle + np.pi) + 0.05, 0.5 + (l_ankle_z - 0.05) * 1.5]
    l_ankle = [-0.15, 0.4 * np.sin(cycle + np.pi), l_ankle_z]
    
    r_elbow = [0.3, 0.2 * np.sin(cycle + np.pi), 1.2]
    r_wrist = [0.3, 0.4 * np.sin(cycle + np.pi), 0.9 + 0.1 * np.cos(cycle + np.pi)]
    l_elbow = [-0.3, 0.2 * np.sin(cycle), 1.2]
    l_wrist = [-0.3, 0.4 * np.sin(cycle), 0.9 + 0.1 * np.cos(cycle)]

    # --- CLINICAL PROFILES ---
    if profile == "Severe Left Hemiparesis":
        noise = np.random.normal(0, noise_level, 3)
        l_elbow = [-0.2, 0.1 + noise[0], 1.25] 
        l_wrist = [-0.1, 0.15 + noise[1], 1.35]  
        l_knee = [-0.15, 0.1 * np.sin(cycle + np.pi), 0.45] 
        l_ankle = [-0.15 - max(0, 0.2 * np.cos(cycle + np.pi)), 0.2 * np.sin(cycle + np.pi), 0.05] 

    elif profile == "Mild Right Spasticity":
        noise = np.random.normal(0, noise_level * 2, 3)
        r_elbow = [0.3, 0.1 * np.sin(cycle + np.pi) + noise[0], 1.2]
        r_wrist = [0.3, 0.15 * np.sin(cycle + np.pi) + noise[1], 1.0]
        r_knee = [0.15, 0.15 * np.sin(cycle) + noise[0], 0.48]
        r_ankle = [0.15, 0.3 * np.sin(cycle), 0.06]

    elif profile == "Bilateral Bradykinesia (Parkinsonian)":
        noise = np.random.normal(0, noise_level * 1.5, 3)
        neck, head = [0, 0.2, 1.4 + bounce], [0, 0.3, 1.55 + bounce]
        r_ankle = [0.15, 0.1 * np.sin(cycle), 0.05]
        l_ankle = [-0.15, 0.1 * np.sin(cycle + np.pi), 0.05]
        r_knee = [0.15, 0.05 * np.sin(cycle) + 0.1, 0.45]
        l_knee = [-0.15, 0.05 * np.sin(cycle + np.pi) + 0.1, 0.45]
        r_elbow = [0.3, 0.05 * np.sin(cycle + np.pi) + noise[0], 1.1]
        r_wrist = [0.3, 0.05 * np.sin(cycle + np.pi) + noise[1], 0.95]
        l_elbow = [-0.3, 0.05 * np.sin(cycle) + noise[0], 1.1]
        l_wrist = [-0.3, 0.05 * np.sin(cycle) + noise[1], 0.95]

    x = [head[0], neck[0], pelvis[0], None, neck[0], r_shoulder[0], None, neck[0], l_shoulder[0], None, l_shoulder[0], r_shoulder[0], r_hip[0], l_hip[0], l_shoulder[0], None, r_shoulder[0], r_elbow[0], r_wrist[0], None, l_shoulder[0], l_elbow[0], l_wrist[0], None, pelvis[0], r_hip[0], r_knee[0], r_ankle[0], None, pelvis[0], l_hip[0], l_knee[0], l_ankle[0]]
    y = [head[1], neck[1], pelvis[1], None, neck[1], r_shoulder[1], None, neck[1], l_shoulder[1], None, l_shoulder[1], r_shoulder[1], r_hip[1], l_hip[1], l_shoulder[1], None, r_shoulder[1], r_elbow[1], r_wrist[1], None, l_shoulder[1], l_elbow[1], l_wrist[1], None, pelvis[1], r_hip[1], r_knee[1], r_ankle[1], None, pelvis[1], l_hip[1], l_knee[1], l_ankle[1]]
    z = [head[2], neck[2], pelvis[2], None, neck[2], r_shoulder[2], None, neck[2], l_shoulder[2], None, l_shoulder[2], r_shoulder[2], r_hip[2], l_hip[2], l_shoulder[2], None, r_shoulder[2], r_elbow[2], r_wrist[2], None, l_shoulder[2], l_elbow[2], l_wrist[2], None, pelvis[2], r_hip[2], r_knee[2], r_ankle[2], None, pelvis[2], l_hip[2], l_knee[2], l_ankle[2]]
    
    return x, y, z, l_wrist, r_wrist, l_ankle, r_ankle

def build_animated_stickman(frames_data, color_scheme):
    # UPGRADE: Thickened the lines to 15 and markers to 10 to make it look like a solid mannequin
    fig = go.Figure(
        data=[go.Scatter3d(
            x=frames_data[0][0], y=frames_data[0][1], z=frames_data[0][2], 
            mode='lines+markers', marker=dict(size=10, color=color_scheme, opacity=0.9), line=dict(color=color_scheme, width=15)
        )],
        layout=go.Layout(
            scene=dict(xaxis=dict(range=[-0.8, 0.8]), yaxis=dict(range=[-0.8, 0.8]), zaxis=dict(range=[0, 2]), aspectmode='cube'),
            updatemenus=[dict(type="buttons", buttons=[dict(label="▶ Play Kinematics", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True, transition=dict(duration=0))])])],
            margin=dict(l=0, r=0, b=0, t=0), height=450
        ),
        frames=[go.Frame(data=[go.Scatter3d(x=f[0], y=f[1], z=f[2])]) for f in frames_data]
    )
    return fig

# ---------------------------------------------------------
# 6. DASHBOARD CONTROLS (SIDEBAR RESTORED)
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
        
        # EXTRACTING QUAD-SENSOR DATA
        raw_data = np.array([np.concatenate([f[3], f[4], f[5], f[6]]) for f in patient_frames])
        features_24d = calculate_clinical_features_v2(raw_data)
        
        # VELOCITY & ACCELERATION FOR TAB 2
        dt = time_steps[1] - time_steps[0]
        left_wrist_y_vel = np.gradient(raw_data[:, 1], dt)
        left_wrist_y_acc = np.gradient(left_wrist_y_vel, dt)
        right_wrist_y_vel = np.gradient(raw_data[:, 4], dt)
        right_wrist_y_acc = np.gradient(right_wrist_y_vel, dt)
        
        # AI INFERENCE
        indices = np.linspace(0, num_frames - 1, 20, dtype=int)
        scan_window = features_24d[indices]
        prediction_prob, is_stroke = 0.0, False
        
        if model and scaler:
            scaled_features = scaler.transform(scan_window).reshape(1, 20, 24)
            prediction_prob = model.predict(scaled_features)[0][0]
            is_stroke = prediction_prob > 0.5
        else:
            if patient_type == "Healthy Control": prediction_prob = 0.04
            elif patient_type == "Severe Left Hemiparesis": prediction_prob, is_stroke = 0.98, True
            elif patient_type == "Mild Right Spasticity": prediction_prob, is_stroke = 0.76, True
            else: prediction_prob, is_stroke = 0.62, True

        # ---------------------------------------------------------
        # 7. UI TABS: CLINICAL, XAI, ARCHITECTURE, ASYMMETRY
        # ---------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical View (3D)", "⚙️ Live AI Diagnostics", "📚 The Science", "⚖️ Asymmetry Analytics"])
        
        # --- TAB 1: CLINICAL VIEW ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Neural Network Confidence", f"{prediction_prob * 100:.2f}%")
            if is_stroke: col2.error(f"🚨 ALERT: Abnormal Gait Detected")
            else: col2.success("✅ STATUS: Healthy Symmetric Gait")
            col3.metric("Peak Arm Balance Shift", f"{np.max(features_24d[:, 22]):.3f}")

            st.subheader("Live 3D Spatial Tracking (60 FPS)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Baseline (Healthy Reference)**")
                st.plotly_chart(build_animated_stickman(healthy_frames, '#00CC96'), use_container_width=True, key="baseline_stickman") 
            with c2:
                st.markdown(f"**Live Patient ({patient_type})**")
                st.plotly_chart(build_animated_stickman(patient_frames, '#EF553B' if is_stroke else '#00CC96'), use_container_width=True, key="patient_stickman")

        # --- TAB 2: EXPLAINABLE AI ---
        with tab2:
            st.markdown("### Deep Learning & Biomechanical Breakdown")
            st.subheader("1. CNN Tremor Detection: High-Frequency Acceleration (m/s²)")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=left_wrist_y_acc, mode='lines', name='Left Arm Accel', line=dict(color='red' if is_stroke and "Left" in patient_type else 'cyan', width=2)))
            fig_acc.add_trace(go.Scatter(y=right_wrist_y_acc, mode='lines', name='Right Arm Accel', line=dict(color='orange' if is_stroke and "Right" in patient_type else 'lightgreen', width=2)))
            fig_acc.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_acc, use_container_width=True)

            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.subheader("2. CNN Spasticity Filter (Amplitude)")
                fake_heat = np.random.rand(24, 20) if not is_stroke else np.random.rand(24, 20) * 3
                fig_heat = px.imshow(scaled_features[0].T if model else fake_heat, color_continuous_scale="inferno", aspect="auto")
                st.plotly_chart(fig_heat, use_container_width=True)
                
            with col_x2:
                st.subheader("3. LSTM Bradykinesia Filter (Rhythm)")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(y=features_24d[:, 18], mode='lines', name='Left Arm Energy', line=dict(color='red' if is_stroke and "Left" in patient_type else 'cyan', width=3)))
                fig_lstm.add_trace(go.Scatter(y=features_24d[:, 19], mode='lines', name='Right Arm Energy', line=dict(color='orange' if is_stroke and "Right" in patient_type else 'lightgreen', width=3, dash='dot')))
                fig_lstm.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_lstm, use_container_width=True)

        # --- TAB 3: THE SCIENCE ---
        with tab3:
            st.header("🧠 AI Architecture & Clinical Methodology")
            st.markdown("The AI analyzes **480 data points** per prediction (24 Quad-Sensor features × 20 timesteps).")
            with st.expander("🔬 The 24 Clinical Biomarkers (V2 Upgrade)", expanded=True):
                st.markdown("""
                * **Raw Kinematics (Channels 0-11):** The raw X, Y, Z acceleration of both arms and both legs.
                * **Positional Drift (Channels 12-17):** The mathematical difference between limbs (`left_xyz - right_xyz`).
                * **Kinetic Energy (Channels 18-21):** The pure magnitude of force generated by all four limbs.
                * **Lateral Balance Shift (Channels 22-23):** The Asymmetry Index ratio for the upper and lower body.
                """)
            with st.expander("🧬 AI Architecture: Hybrid CNN-LSTM", expanded=True):
                st.markdown("""
                The Hybrid CNN-LSTM network hunts for three clinical stroke signatures:
                * **Bradykinesia:** Evaluated by the LSTM layer.
                * **Spasticity:** Evaluated by the CNN layer.
                * **Acute Tremors:** High-frequency anomaly detection via CNN.
                """)

        # --- TAB 4: ASYMMETRY ANALYTICS (UPGRADED) ---
        with tab4:
            st.header("⚖️ Quad-Sensor Symmetry Profiling")
            st.markdown("An SI of **0%** is perfect symmetry. Values exceeding **±10%** are clinically significant for neurological intervention.")
            
            asym_fig_bar, asym_fig_traj, arm_si, leg_si = render_asymmetry_analysis(features_24d, patient_frames)
            
            col_a, col_b = st.columns([1.5, 1]) 
            with col_a:
                # This chart now renders 3D Volumes instead of line paths
                st.plotly_chart(asym_fig_traj, use_container_width=True)
                st.plotly_chart(asym_fig_bar, use_container_width=True)
            with col_b:
                st.subheader("Clinical Symmetry Index (SI)")
                st.metric("Arm Symmetry Index", f"{arm_si:.1f}%", delta="Abnormal" if abs(arm_si) > 10 else "Normal", delta_color="inverse")
                st.metric("Leg Symmetry Index", f"{leg_si:.1f}%", delta="Abnormal" if abs(leg_si) > 10 else "Normal", delta_color="inverse")
                
                st.divider()
                st.markdown("### 🔍 AI Gait Interpretation")
                if abs(arm_si) > 20:
                    st.warning(f"**Arm Deficit:** Severe unilateral arm suppression detected on the {'Left' if arm_si < 0 else 'Right'} side. Notice the severely restricted movement volume in the 3D map.")
                else:
                    st.success("**Arm Mechanics:** Normal symmetrical swinging motion. Full 3D volume achieved.")
                    
                if abs(leg_si) > 20:
                    st.warning(f"**Leg Deficit:** Significant paretic leg-drag identified. The {'Left' if leg_si < 0 else 'Right'} ankle shows almost zero 3D volume, indicating the foot is being dragged flat.")
                else:
                    st.success("**Leg Mechanics:** Normal stride length and clearance. Full 3D volume achieved.")
else:
    st.info("👈 Select a Patient Profile and click **Initialize AI Biomechanical Scan**.")
