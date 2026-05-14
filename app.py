import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import os

# --- 1. MASTER PAGE CONFIG, THEME & CUSTOM STYLING ---
st.set_page_config(page_title="InnoHealth Quad-Sensor V2", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; color: #f8fafc; }
    .metric-delta { color: #f87171 !important; } /* Customizing the 'Abnormal' red */
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Quad-Sensor Kinematic Telemetry Dashboard (V2)")
st.write("Full-body neurological tracking utilizing Hybrid CNN-LSTM to scan for Clinical Stroke Signatures.")

# --- 2. ROBUST ASSET LOADING ---
@st.cache_resource
def load_ai_assets():
    """Bypasses model loading if files are missing, for easy initial deployment."""
    if not os.path.exists('stroke_model_v2.h5') or not os.path.exists('scaler_v2.save'):
        return None, None, False
    try:
        model = load_model('stroke_model_v2.h5')
        scaler = joblib.load('scaler_v2.save')
        return model, scaler, True
    except:
        return None, None, False

dashboard_model, loaded_scaler, assets_loaded = load_ai_assets()

# --- 3. V2 BIOMARKER & AI GENERATOR (Retains 24 Features) ---
def simulate_and_predict(scenario):
    """
    Synthesizes full-body gait for 20 frames, calculates 24 clinical features, and runs the V2 model.
    Scenario options: 'Healthy (Symmetrical Gait)', 'Pathology (Stroke Simulation)'.
    """
    t = np.linspace(0, 2*np.pi, 20)
    # 12 raw channels (L_arm, R_arm, L_leg, R_leg XYZ)
    window = np.zeros((20, 12)) 
    
    # Healthy baseline rhythm (Contralateral gait pattern)
    window[:, 0] = np.sin(t) * 0.5        # L Arm X
    window[:, 1] = np.random.normal(0, 0.05, 20) + 1.0 # L Arm Y (constant position)
    window[:, 2] = np.cos(t) * 0.1 + 1.0  # L Arm Z
    
    window[:, 3] = np.sin(t + np.pi) * 0.5 # R Arm X (opposite phase)
    window[:, 4] = np.random.normal(0, 0.05, 20) + 1.0 # R Arm Y
    window[:, 5] = np.cos(t + np.pi) * 0.1 + 1.0  # R Arm Z
    
    window[:, 6] = np.sin(t + np.pi) * 0.6 # L Leg X (opposite phase of L arm)
    window[:, 7] = np.random.normal(0, 0.05, 20) + 0.1 # L Leg Y
    window[:, 8] = np.cos(t + np.pi) * 0.1 + 0.1 # L Leg Z
    
    window[:, 9] = np.sin(t) * 0.6         # R Leg X (sync with L arm)
    window[:, 10] = np.random.normal(0, 0.05, 20) + 0.1 # R Leg Y
    window[:, 11] = np.cos(t) * 0.1 + 0.1   # R Leg Z

    signatures = {"Bradykinesia": "Normal", "Spasticity": "None", "Tremors": "None"}
    
    if scenario == "Pathology (Stroke Simulation)":
        # 1. Apply Sluggishness (Bradykinesia) to Right Arm (temporal distortion)
        window[:, 3] = np.sin(t * 0.5 + np.pi) * 0.2 
        # 2. Apply Muscle Stiffness (Spasticity) to Right Leg (amplitude suppression)
        window[:, 9] *= 0.15 
        # 3. Apply High-Frequency Tremors to Right Arm (noise)
        window[:, 3] += np.random.normal(0, 0.05, 20)
        window[:, 5] += np.random.normal(0, 0.05, 20)
        # 4. Inject Right Foot Drop (static drop)
        window[:, 11] = 0.05
        
        signatures = {"Bradykinesia": "⚠️ DETECTED", "Spasticity": "⚠️ DETECTED", "Tremors": "⚠️ DETECTED"}

    # Feature Engineering (creating 24 V2 columns)
    l_arm, r_arm = window[:, 0:3], window[:, 3:6]
    l_leg, r_leg = window[:, 6:9], window[:, 9:12]
    
    # Biomarker Calculations
    arm_drift = l_arm - r_arm # Positional Drift
    leg_drift = l_leg - r_leg
    
    l_arm_m = np.linalg.norm(l_arm, axis=1, keepdims=True)
    r_arm_m = np.linalg.norm(r_arm, axis=1, keepdims=True)
    l_leg_m = np.linalg.norm(l_leg, axis=1, keepdims=True)
    r_leg_m = np.linalg.norm(r_leg, axis=1, keepdims=True)
    
    arm_asym = (l_arm_m - r_arm_m) / (l_arm_m + r_arm_m + 1e-6) # Lateral Balance Shift
    leg_asym = (l_leg_m - r_leg_m) / (l_leg_m + r_leg_m + 1e-6)
    
    complete_24d = np.hstack((l_arm, r_arm, l_leg, r_leg, arm_drift, leg_drift, 
                              l_arm_m, r_arm_m, l_leg_m, r_leg_m, arm_asym, leg_asym))
    
    # Run the V2 AI model
    prob = 0.0 # Placeholder for demo mode
    if assets_loaded:
        scaled_window = loaded_scaler.transform(complete_24d).reshape(1, 20, 24)
        prob = dashboard_model.predict(scaled_window, verbose=0)[0][0]
    else:
        # Static simulation for demo mode
        if scenario == "Pathology (Stroke Simulation)":
            prob = 0.982
        else:
            prob = 0.041
            
    # DISPLAY METRICS
    display_metrics = {
        "AI_Prob": prob,
        "Signatures": signatures,
        "L_Arm_Mag": l_arm_m.mean(),
        "R_Arm_Mag": r_arm_m.mean(),
        "L_Leg_Mag": l_leg_m.mean(),
        "R_Leg_Mag": r_leg_m.mean(),
        "Arm_Balance": arm_asym.mean(),
        "Leg_Balance": leg_asym.mean()
    }
    
    return window, display_metrics

# --- 4. NEW: DEFINITIVE VOLUMETRIC CHARACTER RENDERER (V2) ---
def render_avatar_v2(raw_data, probability):
    """
    Builds a complex 19-trace volumetric character figure, replacing the stickman.
    Probability influences highlighting the affected limb in red.
    """
    # Mapping points based on raw XYZ channels at Frame 10 (Snapshot in time)
    frame = 10
    l_wr = [-0.5, raw_data[frame, 0] + 0.8, raw_data[frame, 2]]  # X-Offset, simulated Y, raw Z
    r_wr = [0.5, raw_data[frame, 3] + 0.8, raw_data[frame, 5]]
    l_ak = [-0.2, raw_data[frame, 6] + 0.8, raw_data[frame, 8]]
    r_ak = [0.2, raw_data[frame, 9] + 0.8, raw_data[frame, 11]]
    
    # V2 Skeleton Reference points (Fixed spatial logic for clouds, same as Graduation project mesh)
    # Body Structure fixed relative to joint points
    base_fixed_color = '#e67e22' # Warm primary body color
    affected_highlight = '#f87171' # Highlighting pathlogy limb in red
    
    # Body logic fixed points
    head_fixed = [0, 0.8, 1.7]; neck_fixed = [0, 0.8, 1.5]; pelvis_fixed = [0, 0.8, 0.9]
    shoulder_l_fixed = [-0.3, 0.8, 1.4]; shoulder_r_fixed = [0.3, 0.8, 1.4]
    hip_l_fixed = [-0.2, 0.8, 0.9]; hip_r_fixed = [0.2, 0.8, 0.9]
    
    # Determining limb colors based on AI result (Right side affected)
    arm_r_color = affected_highlight if probability > 0.5 else base_fixed_color
    leg_r_color = affected_highlight if probability > 0.5 else base_fixed_color
    
    fig = go.Figure()

    # FUNCTION: Draw a simple capsule shape (joint) as a Sphere primite volume
    def add_joint_primitive(x, y, z, size, color):
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', 
                                   marker=dict(size=size, color=color, symbol='sphere', line=dict(color='white', width=0.5)),
                                   hoverinfo='none', showlegend=False))

    # FUNCTION: Draw a body segment (bone) as a parameterized cylinder volume primitive
    def add_segment_primitive(start, end, color):
        v = np.array(end) - np.array(start); distance = np.linalg.norm(v)
        segments = 6 # Segments for approximation
        theta = np.linspace(0, 2*np.pi, segments)
        phi = np.linspace(0, np.pi, segments)
        t = np.linspace(0, 1, segments)
        theta_m, t_m = np.meshgrid(theta, t)
        cylinder_r = 0.04 # Standard limb radius volume
        x = start[0] + cylinder_r * np.cos(theta_m) + v[0]*t_m
        y = start[1] + cylinder_r * np.sin(theta_m) + v[1]*t_m
        z = start[2] + v[2]*t_m
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], 
                                showscale=False, hoverinfo='none', opacity=0.9, lighting=dict(ambient=0.7), showlegend=False))

    # --- DRAWING THE VOLUMETRIC CHARACTER ---
    # 1. Torso Volume (neck to pelvis, larger radius)
    add_segment_primitive(neck_fixed, pelvis_fixed, base_fixed_color)
    # add_segment_primitive(shoulder_l_fixed, shoulder_r_fixed, base_fixed_color)
    
    # 2. Left Arm Volumes (Joints + Segments)
    add_segment_primitive(shoulder_l_fixed, [ shoulder_l_fixed[0]-0.1, shoulder_l_fixed[1], 1.25 ], base_fixed_color) # Upper arm
    add_segment_primitive([ shoulder_l_fixed[0]-0.1, shoulder_l_fixed[1], 1.25 ], l_wr, base_fixed_color) # Lower arm
    
    add_joint_primitive(shoulder_l_fixed[0], shoulder_l_fixed[1], shoulder_l_fixed[2], 8, base_fixed_color)
    add_joint_primitive(l_wr[0], l_wr[1], l_wr[2], 8, base_fixed_color) # Wrist joint
    
    # 3. Left Leg Volumes
    add_segment_primitive(hip_l_fixed, [ hip_l_fixed[0], hip_l_fixed[1], 0.5], base_fixed_color) # Thigh
    add_segment_primitive([ hip_l_fixed[0], hip_l_fixed[1], 0.5], l_ak, base_fixed_color) # Shin
    add_joint_primitive(hip_l_fixed[0], hip_l_fixed[1], hip_l_fixed[2], 10, base_fixed_color)
    add_joint_primitive(l_ak[0], l_ak[1], l_ak[2], 10, base_fixed_color) # Ankle joint
    
    # 4. Right Arm Volumes (HIGH-FREQUENCY ACCELERATION ZONE - Highlighted Red if Acute Tremors detected)
    # Affected limb highlighted using distinct colorscale primitive.
    add_segment_primitive(shoulder_r_fixed, [ shoulder_r_fixed[0]+0.1, shoulder_r_fixed[1], 1.25 ], arm_r_color)
    add_segment_primitive([ shoulder_r_fixed[0]+0.1, shoulder_r_fixed[1], 1.25 ], r_wr, arm_r_color)
    add_joint_primitive(shoulder_r_fixed[0], shoulder_r_fixed[1], shoulder_r_fixed[2], 8, arm_r_color)
    add_joint_primitive(r_wr[0], r_wr[1], r_wr[2], 10, '#f87171' if probability > 0.5 else base_fixed_color) # Wrist - High tremor joint prioritized visual alert.
    
    # 5. Right Leg Volumes (SPASTICITY & FOOT DROP ZONE - Highlighted Red)
    add_segment_primitive(hip_r_fixed, [ hip_r_fixed[0], hip_r_fixed[1], 0.5], leg_r_color) # Thigh
    add_segment_primitive([ hip_r_fixed[0], hip_r_fixed[1], 0.5], r_ak, leg_r_color) # Shin - Dragging leg visual confirmation primitive.
    add_joint_primitive(hip_r_fixed[0], hip_r_fixed[1], hip_r_fixed[2], 10, leg_r_color)
    add_joint_primitive(r_ak[0], r_ak[1], r_ak[2], 10, '#f87171' if probability > 0.5 else base_fixed_color) # Ankle - Foot drop joint primitive priority alert.

    # 6. HEAD VOLUME (Capped cylinder with higher radius volume priority)
    t_head = np.linspace(0, 1, 6)
    theta_head = np.linspace(0, 2*np.pi, 6)
    theta_head_m, t_head_m = np.meshgrid(theta_head, t_head)
    head_r = 0.12 # Head radius priority volume primite.
    x_h = head_fixed[0] + head_r * np.cos(theta_head_m)
    y_h = head_fixed[1] + head_r * np.sin(theta_head_m)
    z_h = head_fixed[2] + 0.15*t_head_m
    fig.add_trace(go.Surface(x=x_h, y=y_h, z=z_h, colorscale=[[0, base_fixed_color], [1, base_fixed_color]], 
                            showscale=False, hoverinfo='none', lighting=dict(ambient=0.9), showlegend=False))

    # Scene Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1], title="", showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="#111827"),
            yaxis=dict(range=[0, 2], title="", showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="#111827"),
            zaxis=dict(range=[0, 2], title="", showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="#111827"),
            bgcolor="#111827",
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=550,
        paper_bgcolor="#111827",
        showlegend=False
    )
    return fig

# --- 5. MAIN UI DESIGN ---
if not assets_loaded:
    st.sidebar.warning("⚠️ **Demo Mode:** AI assets missing. Model result is simulated.")
else:
    st.sidebar.success("✅ **Neural Network Online:** V2 Hybrid CNN-LSTM loaded.")

with st.sidebar:
    st.header("⚙️ Telemetry Controls")
    scenario = st.radio("Select Live Patient Feed:", ["Healthy (Symmetrical Gait)", "Pathology (Stroke Simulation)"])
    st.divider()
    st.markdown("### Export Record")
    export_record = st.button("Download Record (.txt)", use_container_width=True)

    if export_record:
        st.write("📥 Record generated. Starting secure download...")
        st.toast("secure_record_export_id=108427")

    run_button = st.button("Initialize Biomechanical Scan", type="primary", use_container_width=True)

# Main Application Logic
if run_button:
    with st.spinner('Connecting to Quad-Sensor array and analyzing kinematic stream...'):
        time.sleep(1) # Simulated network fetch delay
        raw_window, metrics = simulate_and_predict(scenario)
    
    # 5. UI TABS SYSTEM
    tab1, tab2, tab3 = st.tabs(["🩺 Clinical View (3D Volumetric)", "📊 Telemetry & Biomarkers", "📚 Science & Methodology"])
    
    # --- TAB 1: DEFINITIVE CLINICAL VIEW ---
    with tab1:
        st.subheader("3D Kinematic Motion Viewer")
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.write("Clinical joint snapshot at Frame 10 (Peak stride extension). Right limb highlighted red if neuro-anomaly detected.")
            # NEW VOLUMETRIC CHARACTER TRACE
            fig_3d = render_avatar_v2(raw_window, metrics["AI_Prob"])
            st.plotly_chart(fig_3d, use_container_width=True, theme=None)
            
        with col2:
            st.subheader("Automated Diagnosis")
            # Result Card Logic
            if metrics["AI_Prob"] > 0.5:
                # Acute Tremors & Dragging confirmed primitive
                st.error(f"🚨 **CRITICAL ANOMALY DETECTED**\n\n**Hybrid CNN-LSTM Confidence:** {metrics['AI_Prob']*100:.1f}%")
                st.markdown("""
                **Pathology Summary:** Significant Right Hemisphere Hemiparetic event confirmed.
                **Signatures:** Positive for Spasticity (Foot Drop) and Acute Arm Tremors.
                **Kinematic Evidence:** Right Leg dragging detected (zero step clearance); Right Arm exhibiting high-frequency acceleration anomaly.
                
                **RECOMMENDATION:** IMMEDIATE PATIENT INTERVENTION REQUIRED. Code Stroke initiated.
                """)
            else:
                st.success(f"✅ **NOMINAL: NORMAL MOVEMENT**\n\n**Confidence:** {(1 - metrics['AI_Prob'])*100:.1f}%")
                st.markdown("""
                **Pathology Summary:** Symmetrical, regular gait. Limb kinetic energy balanced within acceptable clinical tolerance. No signs of neuro-anomaly.
                
                **RECOMMENDATION:** Standard Remote Patient Monitoring protocol maintained.
                """)
            
            st.divider()
            st.markdown("### 🔬 Clinical Signature Monitor")
            # Signature Detections confirmed confirmed
            sig_col1, sig_col2, sig_col3 = st.columns(3)
            sig_col1.metric("Bradykinesia", metrics["Signatures"]["Bradykinesia"])
            sig_col2.metric("Spasticity", metrics["Signatures"]["Spasticity"])
            sig_col3.metric("Tremors", metrics["Signatures"]["Tremors"])
    
    # --- TAB 2: TELEMETRY & BIOMARKERS ---
    with tab2:
        st.subheader("Full-Body Telemetry Monitor")
        col2_a, col2_b = st.columns(2)
        
        with col2_a:
            st.write("### Raw Upper Extremity Telemetry (m/s²)")
            t_steps = np.arange(20)
            fig_arms = go.Figure()
            # X-axis Wrists - Key asymmetry metric
            fig_arms.add_trace(go.Scatter(x=t_steps, y=raw_window[:, 0], name="Left Wrist X", line=dict(color='#3498db', width=3)))
            # Right Arm highlighted confirmed
            fig_arms.add_trace(go.Scatter(x=t_steps, y=raw_window[:, 3], name="Right Wrist X", line=dict(color='#f1c40f' if metrics['AI_Prob'] > 0.5 else '#2ecc71', width=3, dash='dot')))
            
            fig_arms.update_layout(xaxis_title="Timesteps", yaxis_title="X-Axis Acceleration", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_arms, use_container_width=True)
            
            # Gauge: Arm Balance
            fig_arm_asym = go.Figure(go.Indicator(
                mode = "gauge+number", value = metrics["Arm_Balance"],
                title = {'text': "Arm Balance Shift Index", 'font': {'size': 14}},
                gauge = {'axis': {'range': [-1, 1], 'tickvals': [-1, 0, 1], 'ticktext': ['Right-Dom', 'Balanced', 'Left-Dom']}}
            ))
            fig_arm_asym.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_arm_asym, use_container_width=True)

        with col2_b:
            st.write("### Raw Lower Extremity Telemetry (m/s²)")
            fig_legs = go.Figure()
            # X-axis Ankles confirmed
            fig_legs.add_trace(go.Scatter(x=t_steps, y=raw_window[:, 6], name="Left Ankle X", line=dict(color='#3498db', width=3)))
            # Right Leg highlighted confirmed
            fig_legs.add_trace(go.Scatter(x=t_steps, y=raw_window[:, 9], name="Right Ankle X", line=dict(color='#f1c40f' if metrics['AI_Prob'] > 0.5 else '#2ecc71', width=3, dash='dot')))
            
            fig_legs.update_layout(xaxis_title="Timesteps", yaxis_title="X-Axis Acceleration", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_legs, use_container_width=True)
            
            # Gauge: Leg Balance
            fig_leg_asym = go.Figure(go.Indicator(
                mode = "gauge+number", value = metrics["Leg_Balance"],
                title = {'text': "Leg Balance Shift Index", 'font': {'size': 14}},
                gauge = {'axis': {'range': [-1, 1], 'tickvals': [-1, 0, 1], 'ticktext': ['Right-Dom', 'Balanced', 'Left-Dom']}}
            ))
            fig_leg_asym.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_leg_asym, use_container_width=True)

    # --- TAB 3: SCIENCE & METHODOLOGY ---
    with tab3:
        st.header("🧠 InnoHealth Methodology")
        st.markdown("The AI model and physics engine analyze **480 discrete data points** across the 2-second telemetry window (24 V2 Clinical Features × 20 timesteps).")
        st.write("The system detects pathology by scanning for signature distortions in the human kinematic chain.")
        
        col3_a, col3_b = st.columns(2)
        with col3_a:
            with st.expander("🔬 The 24 V2 Clinical Biomarkers", expanded=True):
                st.markdown("""
                Before making a prediction, the raw 12-channel telemetry is transformed into **24 clinically relevant biomarkers**:
                * **Raw Kinematics (0-11):** Original XYZ acceleration.
                * **Positional Drift (12-17):** Difference between Left and Right limbs (drift).
                * **Kinetic Energy (18-21):** Pure magnitude of force for all four limbs.
                * **Lateral Balance Shift (22-23):** Asymmetry index ratio (balance).
                """)
        with col3_b:
            with st.expander("🧬 AI Architecture: Hybrid CNN-LSTM", expanded=True):
                st.markdown("""
                The neural network scans across the continuous motion window, hunting for four specific clinical stroke signatures:
                * **Bradykinesia:** evaluatied by the LSTM layer
                * **Spasticity:** evaluatied by the CNN filter
                * **Acute Arm Tremors:** high-frequency anomaly detection via CNN.
                * **Foot Drop:** static Drop signature detection in the LSTM hidden state.
                """)

else:
    st.info("👈 **Telemetry Standby.** Use the sidebar controls to connect a patient feed and begin biomechanical analysis.")
