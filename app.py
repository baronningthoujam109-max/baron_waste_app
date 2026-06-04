import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

# Class names (MAKE SURE this matches training order exactly)
class_names = ['glass', 'metal', 'paper', 'plastic']

IMG_SIZE = (224, 224)  # change if your model was trained on different size

def preprocess_image(image: Image.Image):
    # Convert to RGB (important for external images)
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(IMG_SIZE)

    # Convert to array
    img_array = np.array(image)

    # Normalize (VERY IMPORTANT for external accuracy)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


st.title("♻️ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_img = preprocess_image(image)

    # Prediction
    predictions = model.predict(processed_img)

    # Softmax safety (if last layer is not softmax)
    probabilities = tf.nn.softmax(predictions[0]).numpy()

    predicted_class = class_names[np.argmax(probabilities)]
    confidence = np.max(probabilities) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all probabilities
    st.write("Class probabilities:")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {probabilities[i]*100:.2f}%")