import tensorflow as tf
from PIL import Image
import numpy as np

def load_model():
    """Load the trained model"""
    # Load the model. Ensure that your model file classfier_1.h5 is in the same directory.
    model = tf.keras.models.load_model("classfier_1.h5")
    return model

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Resize image to match model's expected input dimensions
    target_size = (224, 224)
    image = image.resize(target_size)
    
    # Convert image to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_class(model, preprocessed_image):
    """Make prediction and return class label and confidence"""
    # Get model predictions
    predictions = model.predict(preprocessed_image)
    
    # Get the predicted class index
    predicted_class_index = int(np.argmax(predictions[0]))
    
    # Get the confidence score
    confidence = float(predictions[0][predicted_class_index] * 100)
    
    # Map class index to label (update these labels as per your dataset)
    class_labels = ["Class_0", "Class_1"]
    predicted_class = class_labels[predicted_class_index]
    
    return predicted_class, confidence
