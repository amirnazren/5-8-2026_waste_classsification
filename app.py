import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Garbage Classification Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGE_SIZE = (256, 256)

CLASS_NAMES = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

CLASS_INFO = {
    "battery":      {"emoji": "🔋", "bin": "Hazardous Waste",   "color": "#e74c3c"},
    "biological":   {"emoji": "🍃", "bin": "Organic/Compost",   "color": "#27ae60"},
    "brown-glass":  {"emoji": "🟤", "bin": "Glass Recycling",   "color": "#8B4513"},
    "cardboard":    {"emoji": "📦", "bin": "Paper Recycling",   "color": "#f39c12"},
    "clothes":      {"emoji": "👕", "bin": "Textile Donation",  "color": "#9b59b6"},
    "green-glass":  {"emoji": "🟢", "bin": "Glass Recycling",   "color": "#2ecc71"},
    "metal":        {"emoji": "⚙️",  "bin": "Metal Recycling",   "color": "#95a5a6"},
    "paper":        {"emoji": "📄", "bin": "Paper Recycling",   "color": "#3498db"},
    "plastic":      {"emoji": "🧴", "bin": "Plastic Recycling", "color": "#e67e22"},
    "shoes":        {"emoji": "👟", "bin": "Textile Donation",  "color": "#1abc9c"},
    "trash":        {"emoji": "🗑️", "bin": "General Waste",     "color": "#7f8c8d"},
    "white-glass":  {"emoji": "⬜", "bin": "Glass Recycling",   "color": "#bdc3c7"},
}

# ─────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("garbage_cnn_model.keras")
    return model

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(model, img_array):
    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    return predicted_index, confidence, predictions[0]

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #2ecc71;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 2px solid #2ecc71;
        margin-bottom: 1rem;
    }
    .result-class {
        font-size: 2rem;
        font-weight: 800;
        color: #2ecc71;
    }
    .result-bin {
        font-size: 1.1rem;
        color: #bdc3c7;
        margin-top: 4px;
    }
    .tip-box {
        background-color: #f0fff4;
        border-left: 4px solid #2ecc71;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.95rem;
        color: #2d3436;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/recycling.png", width=80)
    st.markdown("## ♻️ Garbage Classifier")
    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
    This dashboard uses a **CNN deep learning model** to classify waste images into **12 categories** 
    and recommends the correct disposal bin.
    """)
    st.markdown("---")
    st.markdown("### 🗂️ Waste Categories")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **{cls.title()}** → {info['bin']}")
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("""
    - **Architecture**: CNN (3 Conv Blocks)  
    - **Input Size**: 256 × 256 px  
    - **Classes**: 12  
    - **Regularization**: L2 + Dropout  
    """)

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">♻️ Garbage Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a waste image to classify it and get disposal recommendations</div>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model = load_model()

st.success("✅ Model loaded successfully!")
st.markdown("---")

# ─────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a waste image...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of the waste item"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_result:
    st.markdown("### 🔍 Classification Result")
    if uploaded_file:
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            pred_idx, confidence, all_probs = predict(model, img_array)

        pred_class = CLASS_NAMES[pred_idx]
        info = CLASS_INFO[pred_class]

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem;">{info['emoji']}</div>
            <div class="result-class">{pred_class.upper()}</div>
            <div class="result-bin">🗑️ Dispose in: <strong>{info['bin']}</strong></div>
            <br>
            <div style="color:#bdc3c7;">Confidence: <strong style="color:#2ecc71;">{confidence*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": "Confidence %", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": info["color"]},
                "steps": [
                    {"range": [0, 50], "color": "#fadbd8"},
                    {"range": [50, 75], "color": "#fdebd0"},
                    {"range": [75, 100], "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line": {"color": "green", "width": 4},
                    "thickness": 0.75,
                    "value": 80
                }
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("👆 Please upload an image to get started.")

# ─────────────────────────────────────────────
# Probability Distribution Chart
# ─────────────────────────────────────────────
if uploaded_file:
    st.markdown("---")
    st.markdown("### 📊 Prediction Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": [f"{CLASS_INFO[c]['emoji']} {c}" for c in CLASS_NAMES],
        "Probability (%)": [p * 100 for p in all_probs],
        "Color": [CLASS_INFO[c]["color"] for c in CLASS_NAMES]
    }).sort_values("Probability (%)", ascending=True)

    fig_bar = px.bar(
        prob_df,
        x="Probability (%)",
        y="Class",
        orientation="h",
        color="Class",
        color_discrete_sequence=prob_df["Color"].tolist(),
        title="Confidence per Class",
        text=prob_df["Probability (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Probability (%)",
        yaxis_title="",
        margin=dict(l=10, r=60, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Disposal Tips")

    disposal_tips = {
        "battery":      "⚠️ Never throw batteries in general waste. Take them to a certified e-waste or hazardous waste collection point.",
        "biological":   "🌱 Compost organic waste or place in your green/organic bin. Great for garden fertilizer!",
        "brown-glass":  "🔵 Rinse glass before recycling. Place in the glass recycling bin — do not mix with other waste.",
        "cardboard":    "📦 Flatten boxes before placing in the paper/cardboard recycling bin. Keep it dry!",
        "clothes":      "👗 Donate wearable clothes to charity or textile collection points. Avoid sending to landfill.",
        "green-glass":  "♻️ Clean and place in glass recycling. Remove caps or lids before recycling.",
        "metal":        "🔩 Rinse metal cans and place in metal recycling. Scrap metal centers also accept larger items.",
        "paper":        "📄 Keep paper dry and clean. Shredded paper can go in compost or paper recycling.",
        "plastic":      "🧴 Check the plastic number (1–7). Most councils accept types 1 & 2. Rinse before recycling.",
        "shoes":        "👟 Donate usable shoes to charity shops or shoe banks. Worn-out shoes can go to textile recyclers.",
        "trash":        "🗑️ This item goes into general waste. Try to reduce this type of waste in the future.",
        "white-glass":  "🫙 Place in the glass recycling bin. Avoid mixing with other types of waste.",
    }

    st.markdown(f'<div class="tip-box">{disposal_tips[pred_class]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>Built with ❤️ using TensorFlow & Streamlit | Garbage Classification CNN Model</p>",
    unsafe_allow_html=True
)
