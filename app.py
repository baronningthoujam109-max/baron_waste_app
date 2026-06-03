import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# ---------------------------
# Load Model (FIXED INDENTATION)
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

# ---------------------------
# Class labels
# ---------------------------
class_names = ['glass', 'metal', 'paper', 'plastic']

# ---------------------------
# UI
# ---------------------------
st.title("♻️ Waste Classification App designed by Baron Ningthoujam")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction Section
# ---------------------------
if uploaded_file is not None:
    try:
        # Read image safely
        image_bytes = uploaded_file.getvalue()

        st.write("Filename:", uploaded_file.name)
        st.write("Type:", uploaded_file.type)
        st.write("Size:", uploaded_file.size, "bytes")

        if len(image_bytes) == 0:
            st.error("Uploaded file is empty.")
            st.stop()

        # Open image safely
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # ---------------------------
        # Resize (IMPORTANT FIX: 244 → usually 224)
        # ---------------------------
        img = image.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------------------------
        # Prediction
        # ---------------------------
        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        st.success(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        if confidence < 70:
            st.warning("Low confidence prediction. Image may not match training data.")

        # ---------------------------
        # Probabilities
        # ---------------------------
        st.write("### Class Probabilities")

        for cls, prob in zip(class_names, prediction[0]):
            st.write(f"{cls}: {prob * 100:.2f}%")

        df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": prediction[0] * 100
        })

        st.dataframe(df)

    except UnidentifiedImageError:
        st.error("Cannot read this image. Upload a valid JPG/PNG file.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
---
## ♻️ AI-Based Waste Classification

Upload an image of waste material and the AI model will classify it into:

- Glass  
- Metal  
- Paper  
- Plastic  

Developed by Baron Ningthoujam
""")

st.markdown("📂 View Source Code (add GitHub link here)")