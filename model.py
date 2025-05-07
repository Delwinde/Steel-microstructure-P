
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
from PIL import Image
import tensorflow as tf

class SteelClassifier:
    def __init__(self, model_path='classfier_1.h5'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")
            
        try:
            # Define the model architecture
            input_shape = (128, 128, 3)
            inputs = Input(shape=input_shape)
            
            # Convolutional layers
            x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_11')(inputs)
            x = MaxPooling2D((2, 2), name='max_pooling2d_8')(x)
            
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_12')(x)
            x = MaxPooling2D((2, 2), name='max_pooling2d_9')(x)
            
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_13')(x)
            x = MaxPooling2D((2, 2), name='max_pooling2d_10')(x)
            
            # Flatten and Dense layers
            x = Flatten(name='flatten_4')(x)
            x = Dense(64, activation='relu', name='dense_16')(x)
            x = Dropout(0.5, name='dropout_8')(x)
            x = Dense(32, activation='relu', name='dense_17')(x)
            x = Dropout(0.5, name='dropout_9')(x)
            x = Dense(16, activation='relu', name='dense_18')(x)
            outputs = Dense(4, activation='softmax')(x)
            
            # Create the model
            self.model = Model(inputs=inputs, outputs=outputs)
            
            # Load pre-trained weights
            self.model.load_weights(model_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
        self.image_size = (128, 128)
        self.classes = ['Spheroidite', 'Pearlite', 'Martensite', 'Bainite']
        
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def predict(self, image):
        """Make prediction with error handling"""
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
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
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
