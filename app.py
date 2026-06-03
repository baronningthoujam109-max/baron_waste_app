import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

st.title("♻️ Waste Classification App")
st.write("Upload an image of waste and the model will predict the category.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((244, 244))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    class_names = ['glass', 'metal', 'paper', 'plastic']

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.write("### Class Probabilities")
    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"{cls}: {prob*100:.2f}%")