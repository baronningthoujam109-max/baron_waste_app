import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO

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
try:
# Read image safely
image_bytes = uploaded_file.getvalue()

    st.write("Filename:", uploaded_file.name)
    st.write("Type:", uploaded_file.type)
    st.write("Size:", uploaded_file.size, "bytes")

    if len(image_bytes) == 0:
        st.error("Uploaded file is empty.")
        st.stop()

    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize to model input size
    img = image.resize((244, 244))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    if confidence < 70:
        st.warning(
            "Low confidence prediction. Image may not belong to any trained class."
        )

    st.write("### Class Probabilities")

    for cls, prob in zip(class_names, prediction[0]):
        st.write(f"{cls}: {prob * 100:.2f}%")

    df = pd.DataFrame({
        "Class": class_names,
        "Probability (%)": prediction[0] * 100
    })

    st.dataframe(df)

except UnidentifiedImageError:
    st.error(
        "Cannot read this image. The file may be corrupted or not a valid JPG/PNG."
    )

except Exception as e:
    st.error(f"Unexpected error: {e}")

st.markdown("""

AI-Based Waste Classification

Upload an image of waste material and the AI model will classify it into:

Glass
Metal
Paper
Plastic

Developed by Baron Ningthoujam
""")

st.markdown(
"📂 View Source Code"
)