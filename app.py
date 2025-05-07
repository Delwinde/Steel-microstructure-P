import streamlit as st
from PIL import Image
import numpy as np
from model import load_model, preprocess_image, predict_class

# Set page config
st.set_page_config(
    page_title="Steel Microstructure Classifier",
    page_icon="🔍",
    layout="wide"
)

# Main title
st.title("Steel Microstructure Classification")
st.write("Upload a micrograph image to classify the steel microstructure")

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction when button is clicked
    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            # Preprocess the image
            processed_img = preprocess_image(image)
            
            # Get prediction
            prediction, confidence = predict_class(model, processed_img)
            
            # Display results
            st.success("Classification Complete!")
            st.write(f"Predicted Class: {prediction}")
            st.write(f"Confidence: {confidence:.2f}%")

# Add information about the model
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a deep learning model to classify steel microstructure images. "
    "Upload a micrograph image to get the classification results."
)
