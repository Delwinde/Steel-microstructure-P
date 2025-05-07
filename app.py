
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from model import create_model
import cv2

# Load the trained model
model = create_model()
model.load_weights('classfier_1.h5')

def preprocess_image(image):
    # Resize image to 128x128
    img = cv2.resize(image, (128, 128))
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Steel Microstructure Classification")
    st.write("Upload an image of steel microstructure for classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Preprocess the image
        processed_image = preprocess_image(image_array)
        
        # Make prediction
        if st.button('Classify'):
            prediction = model.predict(processed_image)
            
            # Display result
            st.write("### Classification Result:")
            result = "Class 1" if prediction[0][0] > 0.5 else "Class 0"
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
            st.write(f"Predicted Class: {result}")
            st.write(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()
