import numpy as np  
from tensorflow import keras  
from tensorflow.keras.preprocessing import image  
  
class MicrostructureModel:  
    def __init__(self, model_path):  
        self.model = keras.models.load_model(model_path)  
        self.image_height, self.image_width = 128, 128  # Set the input size  
  
    def load_and_preprocess_image(self, img_path):  
        img = image.load_img(img_path, target_size=(self.image_height, self.image_width))  
        img_array = image.img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension  
        img_array /= 255.0  # Normalize to [0, 1]  
        return img_array  
  
    def predict(self, img_path):  
        preprocessed_image = self.load_and_preprocess_image(img_path)  
        predictions = self.model.predict(preprocessed_image)  
        predicted_class_index = np.argmax(predictions, axis=1)  
        return predicted_class_index[0]  
