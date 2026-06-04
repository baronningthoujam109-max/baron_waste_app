import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

# -----------------------------
# Class Names (MUST match training order)
# -----------------------------
class_names = ['glass', 'metal', 'paper', 'plastic']

# -----------------------------
# App UI
# -----------------------------
st.title("♻️ Waste Classification App")

st.write("Upload an image and the model will predict the waste category.")

# Show model expected shape (DEBUG HELP)
st.write("Model input shape:", model.input_shape)

# -----------------------------
# Safe Image Loader
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    try:
        # Read file safely
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # -----------------------------
        # Dynamic preprocessing (FIXED)
        # -----------------------------
        h, w = model.input_shape[1], model.input_shape[2]

        image = image.resize((w, h))

        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.write("Processed shape:", img_array.shape)

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(img_array)

        probabilities = tf.nn.softmax(predictions[0]).numpy()

        predicted_class = class_names[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.write("Class probabilities:")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {probabilities[i]*100:.2f}%")

    except UnidentifiedImageError:
        st.error("❌ Cannot read image. Please upload a valid JPG or PNG file.")

    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")