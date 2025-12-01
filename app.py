# app.py
import streamlit as st
from PIL import Image
import numpy as np

# local modules (must be in same folder)
from predict import predict, model as loaded_model  # predict returns preds & img_array
from gradcam import generate_gradcam



st.markdown("""
<style>
img:hover{
    transform: scale(1.03);
    transition: 0.3s ease;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# ---- BANNER SECTION ----
banner = Image.open("banner.jpg")

# Make it nicely span the page width
st.image(banner, use_container_width=True)

st.title("🌿 Plant Disease Detector Prototype")
st.write("Upload a tomato leaf image to get a disease prediction and Grad‑CAM heatmap.")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        raise

    # show original (centered in the wide layout columns)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img.resize((300, 300)), caption="Uploaded Image")

    # Predict: returns predicted_class, confidence, preds (raw vector), img_array (preprocessed)
    with st.spinner("Predicting..."):
        predicted_class, confidence, preds, img_array = predict(img)

    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Generate Grad-CAM
    with st.spinner("Generating Grad‑CAM..."):
        class_index = int(np.argmax(preds))
        # pass the preprocessed img_array and the original PIL for overlay
        heatmap_overlay = generate_gradcam(loaded_model, img_array, class_index=class_index, original_image_pil=img)

    st.subheader("Grad‑CAM Heatmap")
    colA, colB = st.columns(2)
    with colA:
        st.image(img.resize((350, 350)), caption="Original Image")
    with colB:
        st.image(heatmap_overlay.resize((350, 350)), caption="Grad‑CAM Heatmap (overlay)")




st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:14px'>
Prototype by Bernice Amponsah • Powered by TensorFlow + Streamlit
</p>
""", unsafe_allow_html=True)
