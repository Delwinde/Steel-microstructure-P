# Import necessary libraries  
import streamlit as st  
import numpy as np  
from keras.models import load_model  
from keras.preprocessing import image  
  
# Load the trained model  
model = load_model('my_model.keras')  
  
# Define the class indices mapping  
class_indices = {0: 'Martensite or Banite', 1: 'Pearlite', 2: 'Similar', 3: 'Spherodized Cementite'}  
  
# Set up the Streamlit app layout  
st.title("Material Classification Model")  
st.write("Upload an image of the material to get a prediction.")  
  
# File uploader for image input  
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])  
  
if uploaded_file is not None:  
    # Load and preprocess the image  
    test_image = image.load_img(uploaded_file, target_size=(128, 128))  
    test_image = image.img_to_array(test_image)  
    test_image = np.expand_dims(test_image, axis=0)  
  
    # Use the loaded model to make the prediction  
    result = model.predict(test_image)  
  
    # Access the predicted class index  
    predicted_class_index = np.argmax(result[0])  
  
    # Get the prediction using the class_indices mapping  
    prediction = class_indices[predicted_class_index]  
  
    # Display the prediction  
    st.write(f"Prediction: {prediction}")  
