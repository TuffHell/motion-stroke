import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="InnoHealth Quad-Sensor Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 Quad-Sensor Kinematic Telemetry Dashboard (V2)")
st.markdown("Real-time full-body neurological tracking utilizing Hybrid CNN-LSTM architecture.")

# --- 2. ROBUST MODEL LOADING ---
@st.cache_resource
def load_ai_assets():
    try:
        model = load_model('stroke_model_v2.h5')
        scaler = joblib.load('scaler_v2.save')
        return model, scaler, True
    except Exception as e:
        return None, None, False

dashboard_model, loaded_scaler, assets_loaded = load_ai_assets()

if not assets_loaded:
    st.error("⚠️ **Model Assets Missing:** Could not find `stroke_model_v2.h5` or `scaler_v2.save`. Please ensure they are in the same directory as this script.")
    st.stop()

# --- 3. 3D SKELETON RENDERER ---
def generate_3d_skeleton(left_wrist, right_wrist, left_ankle, right_ankle):
    """Generates the 3D skeletal figure seen in the screenshots based on the 4 sensor points."""
    # Base fixed points for the torso and head
    head = [0, 0.8, 1.7]; neck = [0, 0.8, 1.5]
    pelvis = [0, 0.8, 0.9]
    l_shoulder = [-0.3, 0.8, 1.4]; r_shoulder = [0.3, 0.8, 1.4]
    l_hip = [-0.2, 0.8, 0.9]; r_hip = [0.2, 0.8, 0.9]
    
    # Calculate elbows and knees based on wrists and ankles (simplified kinematics)
    l_elbow = [-0.4, (l_shoulder[1] + left_wrist[1])/2, 1.2]
    r_elbow = [0.4, (r_shoulder[1] + right_wrist[1])/2, 1.2]
    l_knee = [-0.2, (l_hip[1] + left_ankle[1])/2, 0.5]
    r_knee = [0.2, (r_hip[1] + right_ankle[1])/2, 0.5]

    # Combine into node arrays
    x_nodes = [head[0], neck[0], l_shoulder[0], l_elbow[0], left_wrist[0], l_shoulder[0], r_shoulder[0], r_elbow[0], right_wrist[0], r_shoulder[0], neck[0], pelvis[0], l_hip[0], l_knee[0], left_ankle[0], l_hip[0], r_hip[0], r_knee[0], right_ankle[0]]
    y_nodes = [head[1], neck[1], l_shoulder[1], l_elbow[1], left_wrist[1], l_shoulder[1], r_shoulder[1], r_elbow[1], right_wrist[1], r_shoulder[1], neck[1], pelvis[1], l_hip[1], l_knee[1], left_ankle[1], l_hip[1], r_hip[1], r_knee[1], right_ankle[1]]
    z_nodes = [head[2], neck[2], l_shoulder[2], l_elbow[2], left_wrist[2], l_shoulder[2], r_shoulder[2], r_elbow[2], right_wrist[2], r_shoulder[2], neck[2], pelvis[2], l_hip[2], l_knee[2], left_ankle[2], l_hip[2], r_hip[2], r_knee[2], right_ankle[2]]

    fig = go.Figure()
    # Draw the skeleton lines and joints
    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='lines+markers',
        line=dict(color='#e67e22', width=6),
        marker=dict(size=8, color='#d35400'),
        name="Kinematic Body"
    ))

    # Highlight the 4 sensors
    fig.add_trace(go.Scatter3d(
        x=[left_wrist[0], right_wrist[0], left_ankle[0], right_ankle[0]],
        y=[left_wrist[1], right_wrist[1], left_ankle[1], right_ankle[1]],
        z=[left_wrist[2], right_wrist[2], left_ankle[2], right_ankle[2]],
        mode='markers',
        marker=dict(size=12, color='#2ecc71', symbol='diamond'),
        name="Active Sensors"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1], title="", showgrid=True, backgroundcolor="#111827"),
            yaxis=dict(range=[0, 2], title="", showgrid=True, backgroundcolor="#111827"),
            zaxis=dict(range=[0, 2], title="", showgrid=True, backgroundcolor="#111827"),
            bgcolor="#111827"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
        paper_bgcolor="#111827",
        showlegend=False
    )
    return fig

# --- 4. BIOMARKER & AI GENERATOR ---
def simulate_and_predict(scenario):
    """Simulates a 20-timestep window, calculates the 24 features, and runs the V2 model."""
    window = np.zeros((20, 12)) # 12 raw channels (L_arm, R_arm, L_leg, R_leg XYZ)
    t = np.linspace(0, 2*np.pi, 20)
    
    # Base walking rhythm
    window[:, 0] = np.sin(t) * 0.5  # L Arm X
    window[:, 3] = np.sin(t + np.pi) * 0.5  # R Arm X
    window[:, 6] = np.sin(t + np.pi) * 0.6  # L Leg X
    window[:, 9] = np.sin(t) * 0.6  # R Leg X
    
    if scenario == "Pathology (Stroke Simulation)":
        # Inject Spasticity and Foot Drop on the Right Side
        window[:, 3] *= 0.2 # Arm spasticity
        window[:, 9] *= 0.1 # Shuffling
        window[:, 11] = 0.05 # Z-axis Ankle drop
        true_label = 1
    else:
        true_label = 0

    # Calculate the remaining 12 clinical features (24 total)
    l_arm, r_arm = window[:, 0:3], window[:, 3:6]
    l_leg, r_leg = window[:, 6:9], window[:, 9:12]
    
    arm_drift = l_arm - r_arm
    leg_drift = l_leg - r_leg
    
    l_arm_mag = np.linalg.norm(l_arm, axis=1, keepdims=True)
    r_arm_mag = np.linalg.norm(r_arm, axis=1, keepdims=True)
    l_leg_mag = np.linalg.norm(l_leg, axis=1, keepdims=True)
    r_leg_mag = np.linalg.norm(r_leg, axis=1, keepdims=True)
    
    arm_asym = (l_arm_mag - r_arm_mag) / (l_arm_mag + r_arm_mag + 1e-6)
    leg_asym = (l_leg_mag - r_leg_mag) / (l_leg_mag + r_leg_mag + 1e-6)
    
    complete_24d = np.hstack((l_arm, r_arm, l_leg, r_leg, arm_drift, leg_drift, 
                              l_arm_mag, r_arm_mag, l_leg_mag, r_leg_mag, arm_asym, leg_asym))
    
    # Scale and Predict
    scaled_window = loaded_scaler.transform(complete_24d).reshape(1, 20, 24)
    prob = dashboard_model.predict(scaled_window, verbose=0)[0][0]
    
    # Calculate display metrics
    arm_shift_val = arm_asym.mean()
    leg_shift_val = leg_asym.mean()
    
    # Extract frame 10 for the 3D snapshot
    l_w = [-0.5, window[10, 0] + 0.8, 1.0]
    r_w = [0.5, window[10, 3] + 0.8, 1.0]
    l_a = [-0.2, window[10, 6] + 0.8, 0.1]
    r_a = [0.2, window[10, 9] + 0.8, window[10, 11] + 0.1]
    
    return l_w, r_w, l_a, r_a, arm_shift_val, leg_shift_val, prob

# --- 5. UI LAYOUT ---
with st.sidebar:
    st.header("⚙️ Telemetry Controls")
    scenario = st.radio("Select Live Patient Feed:", ["Healthy (Symmetrical Gait)", "Pathology (Stroke Simulation)"])
    run_button = st.button("Fetch & Analyze Data", type="primary", use_container_width=True)

if run_button:
    with st.spinner('Connecting to Quad-Sensors and running Hybrid CNN-LSTM...'):
        time.sleep(1) # Simulate network fetch
        l_w, r_w, l_a, r_a, arm_shift, leg_shift, ai_confidence = simulate_and_predict(scenario)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("3D Kinematic Motion Viewer")
        fig_3d = generate_3d_skeleton(l_w, r_w, l_a, r_a)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col2:
        st.subheader("AI Diagnosis Engine")
        
        if ai_confidence > 0.5:
            st.error(f"🚨 **CRITICAL ANOMALY DETECTED**\n\n**Pathology Confidence:** {ai_confidence*100:.2f}%")
            st.markdown("*Clinical Note: Severe asymmetry detected. Potential spasticity and foot drop observed on the right side.*")
        else:
            st.success(f"✅ **NOMINAL: NORMAL MOVEMENT**\n\n**Pathology Confidence:** {ai_confidence*100:.2f}%")
            st.markdown("*Clinical Note: Gait is symmetrical. Limb magnitudes are within healthy tolerances.*")
            
        st.divider()
        st.subheader("Clinical Biomarkers")
        
        # Upper Extremity Gauge
        fig_arm = go.Figure(go.Indicator(
            mode = "gauge+number", value = arm_shift,
            title = {'text': "Upper Extremity Balance", 'font': {'size': 14, 'color': 'white'}},
            gauge = {'axis': {'range': [-1.0, 1.0]}, 'bar': {'color': "#3498db" if abs(arm_shift) < 0.2 else "#e74c3c"}}
        ))
        fig_arm.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_arm, use_container_width=True)
        
        # Lower Extremity Gauge
        fig_leg = go.Figure(go.Indicator(
            mode = "gauge+number", value = leg_shift,
            title = {'text': "Lower Extremity Balance", 'font': {'size': 14, 'color': 'white'}},
            gauge = {'axis': {'range': [-1.0, 1.0]}, 'bar': {'color': "#9b59b6" if abs(leg_shift) < 0.2 else "#e67e22"}}
        ))
        fig_leg.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_leg, use_container_width=True)
else:
    st.info("👈 **Awaiting telemetry connection.** Select a scenario in the sidebar and click 'Fetch & Analyze Data' to begin.")
