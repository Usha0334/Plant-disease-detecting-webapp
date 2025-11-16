import streamlit as st
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from predict import predict
import os

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_all_models():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    le = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    svm = pickle.load(open(os.path.join(BASE_DIR, "svm_model.pkl"), "rb"))
    rf = pickle.load(open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb"))

    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

    return le, scaler, svm, rf, base_model


le, scaler, svm, rf, base_model = load_all_models()

# ------------------------------
# Page UI Settings
# ------------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# ------------------------------
# Custom CSS Styling
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #F6FFF8;
}
.title {
    text-align: center;
    color: #2B7A0B;
    font-size: 40px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #52734D;
    font-size: 20px;
}
.result-box {
    padding: 15px;
    background-color: #E3FFE3;
    border-radius: 10px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Title
# ------------------------------
st.markdown("<h1 class='title'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a leaf image and detect the disease instantly.</p>", unsafe_allow_html=True)

# ------------------------------
# Image Upload Section
# ------------------------------
uploaded = st.file_uploader("📁 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded)

    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("Analyzing image..."):
            probs, top_idx = predict(img, le, scaler, rf, svm, base_model)

        st.success("Prediction Complete!")

        # ------------------------------
        # Display Top Prediction
        # ------------------------------
        top_class = le.inverse_transform([top_idx[0]])[0]
        top_prob = probs[top_idx[0]] * 100

        st.markdown(
            f"<div class='result-box'>🌱 <b>Disease:</b> {top_class}<br>📊 <b>Confidence:</b> {top_prob:.2f}%</div>",
            unsafe_allow_html=True
        )

        # ------------------------------
        # Probability Bar Chart
        # ------------------------------
        with col2:
            fig, ax = plt.subplots()
            top5 = top_idx[:5]
            labels = [le.inverse_transform([i])[0] for i in top5]
            values = [probs[i] * 100 for i in top5]

            ax.barh(labels[::-1], values[::-1])
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Top Predictions")

            st.pyplot(fig)

        # ------------------------------
        # Disease Remedies
        # ------------------------------
        remedies = {
            "Apple___Black_rot": [
                "Remove and destroy infected leaves & fruits.",
                "Apply copper-based fungicides.",
                "Improve air circulation around trees.",
                "Avoid overhead watering."
            ],
            "Apple___healthy": [
                "No disease detected!",
                "Maintain proper watering.",
                "Keep leaves clean & dry.",
                "Monitor regularly for symptoms."
            ]
        }

        if top_class in remedies:
            st.markdown("### 🌿 Recommended Remedies")
            for r in remedies[top_class]:
                st.write(f"✔️ {r}")
