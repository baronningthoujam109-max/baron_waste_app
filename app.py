
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

class_names = ['glass', 'metal', 'paper', 'plastic']

st.title("♻️ Waste Classification App designed by Baron Ningthoujam")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

   uploaded_file.seek(0)
image = Image.open(uploaded_file)
image = image.convert("RGB")
from PIL import Image, UnidentifiedImageError

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)
        image = image.convert("RGB")

        st.image(image, caption="Uploaded Image")

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
        st.write("Filename:", uploaded_file.name)
st.write("Type:", uploaded_file.type)
st.write("Size:", uploaded_file.size)
from io import BytesIO

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    st.write("File size:", uploaded_file.size)

    st.image(image, caption="Uploaded Image")

    img = image.resize((244, 244))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.write("### Class Probabilities")

    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"{cls}: {prob*100:.2f}%")

    df = pd.DataFrame({
        "Class": class_names,
        "Probability (%)": prediction[0] * 100
    })

    st.write(df)
    st.markdown("""
### AI-Based Waste Classification

Upload an image of waste material and the AI model will classify it into:

- Glass
- Metal
- Paper
- Plastic

Developed by Baron Ningthoujam
""")
st.markdown(
    "[📂 View Source Code](https://github.com/baronningthoujam109-max/baron_waste_app)"
)