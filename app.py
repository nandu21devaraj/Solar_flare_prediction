import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Solar Flare Detection", layout="centered")

st.title("☀️ Solar Flare Detection System")

st.write("Upload a solar magnetogram (.npy file) to predict flare occurrence")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Solar_Flare_prediction\scripts\solar_flare_model.keras")

model = load_model()

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload .npy file", type=["npy"])

if uploaded_file is not None:

    # load image
# load image 
    img = np.load(uploaded_file) # normalize 
    img = img / np.max(np.abs(img)) 
    input_img = np.expand_dims(img, axis=(0, -1))
    # predict
    pred = model.predict(input_img)[0][0]

    st.subheader("Prediction Result")

    # use true label for demo clarity
    if pred > 0.3:
        st.error(f"⚠️ Solar Flare Detected (Confidence: {pred:.2f})")
    else:
        st.success(f"✅ No Solar Flare (Confidence: {pred:.2f})")

        
    # display image
    st.subheader("Magnetogram")
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)