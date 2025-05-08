
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from model import preprocess_image, get_prediction, class_mapping

st.title('Steel Microstructure Classifier')

uploaded_file = st.file_uploader("Choose a microstructure image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction, confidence = get_prediction(processed_image)
    
    st.write("Predicted Class:", prediction)
    st.write("Confidence: {:.2f}%".format(confidence * 100))
    
    # Display microstructure properties
    st.subheader("Microstructure Properties:")
    st.write("Primary Microconstituent:", class_mapping.get(prediction, "Unknown"))
