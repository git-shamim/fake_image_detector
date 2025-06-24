import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Streamlit config
st.set_page_config(page_title="Fake Image Detector", layout="centered")

# Set paths
MODEL_PATH = "/models/fake_image_model.h5"  # ✅ mounted model path
IMG_SIZE = (224, 224)

# ✅ Load model only once per container (not per request)
@st.cache_resource
def load_model_cached():
    model = load_model(MODEL_PATH)
    return model

model = load_model_cached()

def preprocess_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None

    try:
        img = cv2.resize(img, IMG_SIZE)
    except Exception:
        return None

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    pred = model.predict(image)[0][0]
    label = "Fake" if pred >= 0.5 else "Real"
    confidence = round(pred if label == "Fake" else 1 - pred, 2)
    return label, confidence

# --- Streamlit UI ---
st.title("🕵️ Fake Image Detector")
st.caption("Upload an image to check if it’s real or AI-generated.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = preprocess_image(uploaded_file)

    if image is None:
        st.error("❌ Invalid image. Please upload a valid image file.")
    else:
        label, confidence = predict(image)
        st.success(f"**Prediction: {label}**")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
