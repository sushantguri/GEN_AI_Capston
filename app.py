import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import subprocess
import matplotlib.pyplot as plt

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="NexusYield Emerald Command Center", page_icon="🌾", layout="wide")

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
COLUMNS_PATH = "models/model_columns.pkl"

@st.cache_resource
def load_models():
    m = joblib.load(MODEL_PATH)
    s = joblib.load(SCALER_PATH)
    cols = joblib.load(COLUMNS_PATH)
    return m, s, cols

missing_files = [f for f in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH] if not os.path.exists(f)]
if missing_files:
    st.warning("Initial Setup: Generating ML Models...")
    with st.spinner("Training Random Forest AI..."):
        import sys
        subprocess.run([sys.executable, "src/train_local.py"], check=True)
        st.rerun()

model, scaler, model_columns = load_models()

# --- ADVANCED EMERALD CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
    0% { outline: 0px solid rgba(16, 185, 129, 0.4); }
    50% { outline: 10px solid rgba(16, 185, 129, 0); }
    100% { outline: 0px solid rgba(16, 185, 129, 0); }
}

.stApp {
    background: linear-gradient(-45deg, #09090b, #064e3b, #020617, #022c22);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    font-family: 'Outfit', sans-serif;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.9) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(16, 185, 129, 0.2);
}

h1, h2, h3, h4, h5, p, label, .stMarkdown {
    color: #f8fafc !important;
}

.main-header {
    padding: 2rem;
    text-align: center;
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(25px);
    border-radius: 24px;
    border: 1px solid rgba(168, 224, 99, 0.2);
    box-shadow: 0 15px 40px rgba(0,0,0,0.5);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #bbf7d0, #10b981, #047857);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1.5px;
}

.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.5rem;
}

.metric-box {
    text-align: center;
    padding: 2.5rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 20px;
    border: 2px dashed rgba(16, 185, 129, 0.5);
    animation: pulseGlow 3s infinite;
}

.status-label {
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #a7f3d0;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 4.5rem;
    font-weight: 800;
    color: #6ee7b7;
    text-shadow: 0 0 30px rgba(110, 231, 183, 0.5);
    line-height: 1;
}

.agent-report {
    background: rgba(0, 0, 0, 0.4);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid rgba(52, 211, 153, 0.2);
}

.agent-report h3 {
    color: #34d399 !important;
    border-bottom: 1px solid rgba(52, 211, 153, 0.3);
    padding-bottom: 0.5rem;
    margin-top: 1.5rem;
    font-size: 1.2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: EMERALD CONTROL PANEL ---
with st.sidebar:
    st.markdown("## 🚜 Control Panel")
    
    st.markdown("### 🔑 API Configuration")
    user_api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password", help="Enter your Groq API key to power the Advisory Agent.")
    
    st.markdown("---")
    st.markdown("### 🌍 Farm Parameters")
    soil_type = st.selectbox("Soil Type", ['Chalky', 'Clay', 'Loam', 'Peaty', 'Sandy', 'Silt'])
    crop = st.selectbox("Crop", ['Barley', 'Cotton', 'Maize', 'Rice', 'Soybean', 'Wheat'])
    rainfall = st.slider("Rainfall (mm)", 0.0, 2000.0, 550.0)
    temperature = st.slider("Average Temp (°C)", -10.0, 60.0, 27.5)
    days_to_harvest = st.slider("Days to Harvest", 30, 300, 104)
    
    st.markdown("### 🚜 Infrastructure")
    col_c1, col_c2 = st.columns(2)
    fertilizer = col_c1.checkbox("Fertilizer", value=True)
    irrigation = col_c2.checkbox("Irrigation", value=True)

# --- LIVE PREDICTION LOGIC (DYNAMIC GRAPH) ---
input_data = pd.DataFrame([{
    'Rainfall_mm': rainfall, 'Temperature_Celsius': temperature,
    'Days_to_Harvest': days_to_harvest, 'Fertilizer_Used': int(fertilizer),
    'Irrigation_Used': int(irrigation), 'Soil_Type': soil_type, 'Crop': crop
}])
scaled_features = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
input_data[scaled_features] = scaler.transform(input_data[scaled_features])
input_data = pd.get_dummies(input_data, columns=['Soil_Type', 'Crop']).reindex(columns=model_columns, fill_value=0)

predicted_yield = model.predict(input_data)[0]
farm_params_str = f"Crop: {crop}, Soil: {soil_type}, Rain: {rainfall}mm, Temp: {temperature}C"

# --- MAIN DASHBOARD: EMERALD COMMAND CENTER ---
st.markdown("""
<div class="main-header">
    <h1 class="hero-title">NexusYield Enterprise</h1>
    <p style="color:#94a3b8; margin-top:0.3rem;">Agentic AI Agronomy Command Center</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.4])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Live Predictive Stats")
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="status-label">Live Yield Intensity</div>
        <div class="metric-value">{predicted_yield:.2f} <span style="font-size: 1.5rem; color:#f8fafc;">t/ha</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Dynamic Feature Impact")
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:5]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0) 
    ax.set_facecolor('none')
    ax.barh(range(5), importances[idx], color='#10b981', alpha=0.9, edgecolor='white')
    ax.invert_yaxis()
    ax.set_yticks(range(5))
    ax.set_yticklabels([model_columns[i].replace('_', ' ').replace('Type ', '').title() for i in idx], color='white', fontweight='bold')
    ax.tick_params(axis='x', colors='white')
    for spine in ax.spines.values():
        spine.set_color('none')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Agentic Advisory Stream")
    st.markdown(f"**🟢 Current State:** `{farm_params_str}`")
    
    custom_query = st.text_input("Ask the Agronomist", value="Based on my parameters, what are the top 3 risks and mitigation steps?")
    
    if st.button("Query Advisory Agent"):
        if not user_api_key:
            st.error("Please enter your Groq API Key in the sidebar to use the Advisory Agent.")
        else:
            with st.spinner("Retrieving FAISS Context & Synthesizing Groq Report..."):
                try:
                    from src.agent import run_agentic_workflow
                    report = run_agentic_workflow(
                        farm_data_str=farm_params_str + f" (Predicted Yield: {predicted_yield:.2f} t/ha)", 
                        query=custom_query,
                        api_key=user_api_key
                    )
                    st.markdown(f'<div class="agent-report">{report}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Agent analysis failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
