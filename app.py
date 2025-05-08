
import streamlit as st  
import numpy as np  
from tensorflow import keras  
from tensorflow.keras.preprocessing import image  
  
# Load the trained model  
model = keras.models.load_model('my_model.keras')  
  
# Define preprocessing parameters  
image_height, image_width = 128, 128  
  
# Function to load and preprocess the image  
def load_and_preprocess_image(img_path, target_size):  
    img = image.load_img(img_path, target_size=target_size)  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension  
    img_array /= 255.0  # Normalize to [0, 1]  
    return img_array  
  
# Streamlit app layout  
st.title("Microstructure Classification")  
st.write("Upload an image of a microstructure to classify it.")  
  
# File uploader for image input  
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])  
  
if uploaded_file is not None:  
    # Load and preprocess the image  
    preprocessed_image = load_and_preprocess_image(uploaded_file, (image_height, image_width))  
  
    # Make predictions  
    predictions = model.predict(preprocessed_image)  
  
    # Get the predicted class index  
    predicted_class_index = np.argmax(predictions, axis=1)  
  
    # Map the predicted class index back to the label  
    label_mapping = {0: 'pearlite', 1: 'spheroidite', 2: 'pearlite+spheroidite',  
                     3: 'spheroidite+widmanstatten', 4: 'network', 5: 'martensite'}  
    predicted_label = label_mapping[predicted_class_index[0]]  
  
    # Display the result  
    st.write(f'Predicted class index: {predicted_class_index[0]}')  
    st.write(f'Predicted label: {predicted_label}')  
