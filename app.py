import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Website heading
st.title("🌱 Crop Disease Detection")
st.write("Upload a tomato leaf image to predict its disease.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/crop_model.keras"
    )

model = load_model()

# Load disease names
with open("models/classes.json", "r") as f:
    class_names = json.load(f)

# Image upload
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf", width=300)

    if st.button("Detect Disease"):

        # Resize image
        image = image.resize((128, 128))

        # Convert image to array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)

        # Get predicted class
        predicted_index = np.argmax(prediction[0])

        disease = class_names[predicted_index]

        confidence = float(
            prediction[0][predicted_index] * 100
        )

        # Display result
        st.success("Prediction Completed!")

        st.subheader("Detected Class:")
        st.write(disease)

        st.subheader("Model Confidence:")
        st.write(f"{confidence:.2f}%")

        st.info(
            "This is an educational AI prediction. "
            "Please verify disease identification with an expert."
        )