import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import matplotlib.pyplot as plt
# --- Page Config ---
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")

# --- Try to load models ---
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
COLUMNS_PATH = "models/model_columns.pkl"

missing_files = []
for file in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    # --- Auto-Train Bootstrapper ---
    # Because model.pkl is 3.6GB, it cannot be hosted on GitHub.
    # Therefore, when this app boots in Streamlit Cloud, the file will be missing.
    # We catch the missing file and automatically train the model on the cloud server!
    st.warning(f"Initial Setup: Generating ML Models ({', '.join([os.path.basename(f) for f in missing_files])})")
    
    with st.spinner("Training Random Forest AI... This might take 1-2 minutes for the first boot!"):
        import sys
        try:
            # Run the training script via subprocess so it acts as an independent execution thread
            # capture_output=False allows Streamlit Cloud logs to show the training progress
            # Using sys.executable ensures we use the exact Python environment running Streamlit
            print(f"Running subprocess: {sys.executable} src/train_local.py")
            subprocess.run([sys.executable, "src/train_local.py"], check=True)
            st.success("✅ Agronomy Engine successfully trained and initialized!")
            st.rerun()
        except Exception as e:
            print(f"Subprocess Exception -> {e}")
            st.error("Failed to dynamically train the model.")
            st.code(str(e))
            st.stop()

@st.cache_resource
def load_models():
    m = joblib.load(MODEL_PATH)
    s = joblib.load(SCALER_PATH)
    cols = joblib.load(COLUMNS_PATH)
    return m, s, cols

# Load real models if available
model, scaler, model_columns = load_models()

# --- Custom CSS for Breathtaking UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

/* Hide Defaults */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Background */
.stApp {
    background: linear-gradient(120deg, #020617, #064e3b, #0f172a, #022c22);
    background-size: 300% 300%;
    animation: gradientBG 20s ease infinite;
    font-family: 'Poppins', sans-serif;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}


/* Base Glass Containers applied to native columns */
[data-testid="column"] {
    background: rgba(15, 23, 42, 0.4); /* darker, richer glass opacity */
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 0 0 1px rgba(255, 255, 255, 0.05);
    color: #fff;
    animation: slideUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Remove glass from the spacer column */
[data-testid="column"]:nth-of-type(2) {
    background: transparent;
    border: none;
    box-shadow: none;
    backdrop-filter: none;
    padding: 0;
}

/* Stop nested columns (like inside the form) from doubling up on the styling */
[data-testid="column"] [data-testid="column"] {
    background: transparent;
    border: none;
    box-shadow: none;
    backdrop-filter: none;
    padding: 0;
    animation: none;
}

@keyframes slideUp {
    from { transform: translateY(50px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

/* Typography Overrides */
h1, h2, h3, p, label, .stMarkdown {
    color: #ffffff !important;
}
.hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #ffffff, #d4fc79);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 10px 20px rgba(0,0,0,0.1);
    margin-top: 1.5rem;
    margin-bottom: 0px;
    letter-spacing: -2px;
}
.hero-subtitle {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 300;
    color: rgba(255,255,255,0.9) !important;
    margin-bottom: 3rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    border-bottom: 2px solid rgba(255,255,255,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Inputs styling */
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(0, 0, 0, 0.4) !important;
    border: 2px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px;
    color: white !important;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}
.stSelectbox div[data-baseweb="select"] > div:hover {
    border-color: #A8E063 !important;
    background: rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 0 15px rgba(168, 224, 99, 0.3);
}

/* Sliders */
.stSlider div[data-baseweb="slider"] > div > div > div {
    background: #A8E063 !important; /* track active */
}
.stSlider div[data-baseweb="slider"] div[role="slider"] {
    background-color: #ffffff !important;
    border: 4px solid #56AB2F;
    width: 22px; height: 22px;
    box-shadow: 0 0 15px rgba(86, 171, 47, 0.8);
    transition: transform 0.2s;
}
.stSlider div[data-baseweb="slider"] div[role="slider"]:hover {
    transform: scale(1.4);
}

/* Glowing Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #A8E063 0%, #56AB2F 100%);
    color: white;
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    padding: 1.5rem;
    border: none;
    border-radius: 50px;
    box-shadow: 0 15px 30px rgba(86, 171, 47, 0.4);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    text-transform: uppercase;
    letter-spacing: 3px;
    position: relative;
    overflow: hidden;
    margin-top: 1rem;
}
.stButton>button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
    animation: shine 3s infinite;
}
@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}
.stButton>button:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 25px 40px rgba(86, 171, 47, 0.7);
    color: white;
    border: none;
}

/* Stunning Result Box */
.result-box {
    background: radial-gradient(circle at center, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(168, 224, 99, 0.5);
    border-radius: 30px;
    padding: 4rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1.5rem;
    animation: popIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    box-shadow: inset 0 0 50px rgba(168, 224, 99, 0.1), 0 20px 40px rgba(0,0,0,0.5);
}
@keyframes popIn {
    0% { transform: scale(0.9) translateY(20px); opacity: 0; }
    100% { transform: scale(1) translateY(0); opacity: 1; }
}

/* Crop decorative SVG inside result */
.result-box::before {
    content: '🌾';
    font-size: 250px;
    position: absolute;
    opacity: 0.12;
    top: -60px;
    right: -40px;
    transform: rotate(15deg);
    pointer-events: none;
    filter: blur(2px);
}

.result-value {
    font-size: 6.5rem;
    font-weight: 800;
    background: linear-gradient(to bottom, #ffffff, #d4fc79);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    text-shadow: 0 10px 40px rgba(0,0,0,0.2);
    animation: floatNum 4s ease-in-out infinite;
    position: relative;
    z-index: 2;
}
@keyframes floatNum {
    0% { transform: translateY(0px); filter: drop-shadow(0 0 20px rgba(212,252,121,0.4));}
    50% { transform: translateY(-10px); filter: drop-shadow(0 0 35px rgba(212,252,121,0.8));}
    100% { transform: translateY(0px); filter: drop-shadow(0 0 20px rgba(212,252,121,0.4));}
}
.result-unit {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f1f8e9;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 1rem;
    position: relative;
    z-index: 2;
}

/* Checkbox specific tweaks */
.stCheckbox > div > div > div > div {
    background: rgba(255,255,255,0.2) !important;
    border-color: rgba(255,255,255,0.5) !important;
}
.stCheckbox > div > div > div {
    color: white !important;
}

/* Disable generic chart background */
.stImage, .stPyplot {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.3);
}

/* Feature importance header */
.chart-header {
    text-align: center;
    font-size: 1.8rem;
    margin-bottom: 2rem !important;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #ffffff;
    text-transform: uppercase;
}

/* Diagnostic Cards */
.diagnostic-card {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    flex: 1;
    text-align: center;
    transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
}
.diagnostic-card:hover {
    transform: translateY(-5px);
    background: rgba(0, 0, 0, 0.3);
    border-color: #A8E063;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.diag-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 15px rgba(255,255,255,0.3);
}
.diag-title {
    font-size: 0.85rem;
    color: #bbf7d0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.diag-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
</style>


""", unsafe_allow_html=True)

# --- Define Input Options based on Colab Data ---
soil_types = ['Chalky', 'Clay', 'Loam', 'Peaty', 'Sandy', 'Silt']
crops = ['Barley', 'Cotton', 'Maize', 'Rice', 'Soybean', 'Wheat']

# --- UI Header ---
st.markdown("""
<div class="hero-title">NexusYield</div>
<div class="hero-subtitle">Intelligent Agronomy Engine</div>
""", unsafe_allow_html=True)

# --- UI Layout ---
col1, space, col2 = st.columns([1.2, 0.05, 1])

with col1:
    st.markdown('<div class="section-title">🌍 Environmental Parameters</div>', unsafe_allow_html=True)
    
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        soil_type = st.selectbox("Soil Composition", soil_types)
    with row1_c2:
        crop = st.selectbox("Cultivar Type", crops)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    rainfall = st.slider("Precipitation (mm)", min_value=0.0, max_value=2000.0, value=550.0)
    temperature = st.slider("Mean Temperature (°C)", min_value=-10.0, max_value=60.0, value=27.5)
    days_to_harvest = st.slider("Maturation Window (Days)", min_value=30, max_value=300, value=104)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚜 Intervention Strategy</div>', unsafe_allow_html=True)
    
    row3_c1, row3_c2 = st.columns(2)
    with row3_c1:
        fertilizer = st.checkbox("Chemical Fertilizer Applied", value=True)
    with row3_c2:
        irrigation = st.checkbox("Active Irrigation System", value=True)

with col2:
    predict_clicked = st.button("Simulate Yield")
    result_placeholder = st.empty()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title" style="margin-top:20px;">🧠 System Diagnostics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; gap: 1.5rem; margin-top: 1rem;">
        <div class="diagnostic-card">
            <div class="diag-icon">⚙️</div>
            <div class="diag-title">Engine Architecture</div>
            <div class="diag-value">Ensemble Forest</div>
        </div>
        <div class="diagnostic-card">
            <div class="diag-icon">🎯</div>
            <div class="diag-title">Model Confidence</div>
            <div class="diag-value">91.00% R²</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Prediction Logic ---
if predict_clicked:
    # 1. Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'Rainfall_mm': rainfall,
        'Temperature_Celsius': temperature,
        'Days_to_Harvest': days_to_harvest,
        'Fertilizer_Used': int(fertilizer),
        'Irrigation_Used': int(irrigation),
        'Soil_Type': soil_type,
        'Crop': crop
    }])
    
    # 2. Scale
    features_to_scale = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
    input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])
    
    # 3. Dummies
    input_data = pd.get_dummies(input_data, columns=['Soil_Type', 'Crop'])
    
    # 4. Reindex
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # 5. Predict
    prediction = model.predict(input_data)[0]
    
    # 6. Display Ultra-Premium Result
    result_placeholder.markdown(f"""
    <div class="result-box">
        <div class="result-value">{prediction:.2f}</div>
        <div class="result-unit">Tons / Hectare</div>
    </div>
    """, unsafe_allow_html=True)

    # 7. Sleek Feature Importance
    if hasattr(model, 'feature_importances_'):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-header">Feature Impact Matrix</div>', unsafe_allow_html=True)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5] # Top 5
        
        # Transparent chart with white text
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_alpha(0.0) # Transparent background
        ax.set_facecolor('none')
        
        # Glowing bars
        bars = ax.barh(range(len(indices)), importances[indices], align="center", color='#A8E063', alpha=0.9, height=0.5, zorder=3)
        
        # Invert y axis to have highest on top
        ax.invert_yaxis()
        
        ax.set_yticks(range(len(indices)))
        clean_names = [model_columns[i].replace('_', ' ').replace('Type ', '').title() for i in indices]
        ax.set_yticklabels(clean_names, color='#ffffff', fontsize=12, fontweight='bold')
        
        ax.tick_params(axis='x', colors='#ffffff', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.4))
        ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.4))
        
        # Add a subtle grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
        
        plt.tight_layout()
        st.pyplot(fig)
