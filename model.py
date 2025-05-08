
import tensorflow as tf
import numpy as np
from PIL import Image

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class mapping from your Excel file
class_mapping = {
    "ET Gyro": "pearlite",
    "AC1 800C 85H Q": "spheroidite",
    "SIS_XL.TIF": "quality",
    "AC1 970C 90M FC": "pearlite+spheroidite",
    "AC1 750C 5M Q": "pearlite"
    # Add other mappings as needed
}

def load_model():
    return tf.keras.models.load_model('steel_microstructure_model.h5')

def preprocess_image(image):
    # Resize image
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_prediction(preprocessed_image):
    model = load_model()
    predictions = model.predict(preprocessed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    
    # Map index to class name using your class_indices dictionary
    index_to_class = {
        0: "AC1 800C 85H Q",
        1: "AC1 970 5M FC",
        2: "AC1 970C 24H Q",
        3: "AC1 970C 8H Q",
        4: "AC1 970C 90M 65C 4H FC",
        5: "AC1 970C 90M FC",
        6: "AC1 970C 90M Q",
        7: "AC1 IS1010H 35X etched",
        8: "ET Gyro",
        9: "SIS_XL.TIF"
    }
    
    predicted_class = index_to_class[predicted_index]
    return predicted_class, confidence
