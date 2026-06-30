# CellVue - HealingPro Technologies
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
from tensorflow.keras.applications.efficientnet import preprocess_input

# Branding
BRAND_NAME = "CellVue"
BRAND_SUB = "Diagnostic Command Center | HealingPro Technologies"
BRAND_COLOR = "#0077B6" # Professional Medical Blue
ACCENT_COLOR = "#2ecc71" # Success Green
BG_GRAY = "#F6F8FA"
TEXT_MUTED = "#6B7280"
IMAGE_DISPLAY_WIDTH = 350
IMG_SIZE = 180

# Page config
st.set_page_config(
    page_title=f"{BRAND_NAME} — Clinical Diagnostics",
    page_icon="🔬",
    layout="wide",
)

# CSS Styling
st.markdown(
    f"""
    <style>
    .stApp, .main, .block-container, body {{ background-color: {BG_GRAY} !important; color: #111827 !important; }}
    [data-testid="stSidebar"] {{ background-color: #ffffff !important; }}
    .brand-title {{ color: {BRAND_COLOR} !important; font-size:38px !important; font-weight:700 !important; margin: 0; }}
    .brand-sub {{ color: {TEXT_MUTED} !important; font-size:16px; margin-top:4px; margin-bottom:12px; }}
    .card {{ background: #ffffff !important; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important; border: 1px solid rgba(0,0,0,0.05) !important; }}
    div.stButton > button {{ background-color: {BRAND_COLOR} !important; color: #ffffff !important; border-radius: 8px !important; font-weight: 600 !important; }}
    div.stButton > button:hover {{ filter: brightness(1.1) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Model
MODEL_PATH = "Malaria_Cell_Classification_Model.h5"
CLASS_NAMES = ["Parasitized", "Uninfected"]

# Load model
@st.cache_resource
def load_model(path: str):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        return None

model = load_model(MODEL_PATH)

# Sidebar
with st.sidebar:
    st.markdown(f"# {BRAND_NAME}")
    st.write("Ending the era of presumptive diagnosis. Powered by HealingPro Technologies.")
    st.divider()
    st.markdown("### Clinical Protocol")
    st.write("1. Capture/Upload blood-smear image.")
    st.write("2. AI validates parasitic density.")
    st.write("3. Log result to surveillance portal.")
    st.divider()
    st.write("**Operational Status:**")
    st.success("System Operational" if model is not None else "Model Error")
    st.caption("iDICE Founders Lab | Cohort 1")

# Preprocess image
def preprocess_image(uploaded_file):
    display_img = Image.open(uploaded_file).convert("RGB")
    img = tf.keras.utils.img_to_array(display_img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img_pre = preprocess_input(img)
    img_pre = tf.expand_dims(img_pre, 0)
    return img_pre, display_img

# Header
st.markdown(f"""
    <div class="card" style="margin-bottom:20px;">
        <h1 class="brand-title">🔬 {BRAND_NAME}</h1>
        <div class="brand-sub">{BRAND_SUB}</div>
    </div>
""", unsafe_allow_html=True)

# Main UI
uploaded_file = st.file_uploader("Upload Giemsa-stained blood smear (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    if model is None:
        st.error("Diagnostic engine (model) not found. Please verify the .h5 file path.")
        st.stop()

    img_tensor, display_img = preprocess_image(uploaded_file)
    
    with st.spinner("CellVue AI is analyzing slide..."):
        raw_preds = model.predict(img_tensor)
        # Handle different output shapes
        if raw_preds.ndim == 2 and raw_preds.shape[1] == 2:
            probs = raw_preds[0]
        else:
            probs = [1.0 - raw_preds[0][0], raw_preds[0][0]]
        
        top_index = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[top_index]
        confidence = float(probs[top_index] * 100.0)

    # Result layout
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(display_img, use_column_width=True)
        st.markdown(f"### Result: **{predicted_label}**")
        st.progress(min(max(confidence/100, 0.0), 1.0))
        st.write(f"Confidence: {confidence:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Diagnostic Notes")
        st.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.write("This tool is for screening purposes. Please confirm with facility standard operating procedures.")
        if predicted_label == "Parasitized":
            st.warning("⚠️ Action Required: Parasites detected. Proceed to therapeutic allocation.")
        else:
            st.success("✅ Cell clear. Continue standard observation.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload a slide image to initiate the CellVue diagnostic process.")

# Footer
st.markdown(f"<br><hr><center><small>© {datetime.datetime.now().year} HealingPro Technologies. Built for Public Health Integrity.</small></center>", unsafe_allow_html=True)
