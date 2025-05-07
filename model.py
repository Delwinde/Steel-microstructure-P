
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

class SteelClassifier:
    def __init__(self, model_path='classfier_1.h5'):
        self.model = load_model(model_path)
        self.image_size = (128, 128)
        self.classes = ['Class1', 'Class2', 'Class3', 'Class4']  # Replace with your actual class names
    
    def preprocess_image(self, image):
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(self.image_size)
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image):
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.classes, predictions[0])
            }
        }
