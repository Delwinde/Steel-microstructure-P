# Import necessary libraries  
import streamlit as st  
import numpy as np  
from keras.models import load_model  
from keras.preprocessing import image  
from PIL import Image  
import io  
  
# Load the trained model  
model = load_model('my_model.keras')  
  
# Define the class indices mapping  
class_indices = {  
    0: 'Martensite or Bainite',  
    1: 'Pearlite',  
    2: 'Similar',  
    3: 'Spheroidized Cementite'  
}  
  
# Define detailed information for each microstructure  
microstructure_info = {  
    'Martensite or Bainite': {  
        'description': 'Martensite and Bainite are transformation products of austenite formed during rapid cooling.',  
        'characteristics': [  
            'Martensite: Needle-like or lath-like structure',  
            'Bainite: Feathery or acicular structure',  
            'Both are hard and strong phases'  
        ],  
        'composition': 'Carbon content typically 0.3-1.0% C',  
        'properties': 'High hardness (40-65 HRC), high strength, low ductility',  
        'formation': 'Formed by rapid cooling (quenching) or isothermal transformation'  
    },  
    'Pearlite': {  
        'description': 'Pearlite is a two-phased, lamellar (or layered) structure composed of alternating layers of ferrite and cementite.',  
        'characteristics': [  
            'Lamellar structure',  
            'Moderate hardness and strength',  
            'Good ductility'  
        ],  
        'composition': 'Eutectoid composition (0.8% C)',  
        'properties': 'Moderate hardness (200-300 HV), good ductility',  
        'formation': 'Formed by slow cooling of austenite'  
    },  
    'Similar': {  
        'description': 'Microstructure is similar to the reference image provided.',  
        'characteristics': [  
            'Visual similarity detected',  
            'Further analysis may be required'  
        ],  
        'composition': 'N/A',  
        'properties': 'N/A',  
        'formation': 'N/A'  
    },  
    'Spheroidized Cementite': {  
        'description': 'Spheroidized cementite consists of spherical cementite particles in a ferrite matrix.',  
        'characteristics': [  
            'Spherical cementite particles',  
            'Soft and ductile structure'  
        ],  
        'composition': 'High carbon steel (>0.8% C)',  
        'properties': 'Low hardness, high machinability',  
        'formation': 'Formed by prolonged heating at just below eutectoid temperature'  
    }  
}  
  
# Streamlit app title and description  
st.title('Microstructure Classification App')  
st.write('Upload a microstructure image to classify and get detailed information.')  
  
# File uploader for user to upload an image  
uploaded_file = st.file_uploader("Choose a microstructure image...", type=["jpg", "jpeg", "png"])  
  
if uploaded_file is not None:  
    # Display the uploaded image  
    img = Image.open(uploaded_file)  
    st.image(img, caption='Uploaded Image', use_column_width=True)  
      
    # Preprocess the image for the model  
    img_resized = img.resize((224, 224))  
    img_array = image.img_to_array(img_resized)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  # Normalize  
  
    # Make prediction  
    prediction = model.predict(img_array)  
    predicted_class = np.argmax(prediction, axis=1)[0]  
    predicted_label = class_indices[predicted_class]  
  
    # Show prediction  
    st.subheader('Prediction')  
    st.write('Predicted Microstructure: **' + predicted_label + '**')  
  
    # Show detailed information  
    st.subheader('Microstructure Details')  
    info = microstructure_info[predicted_label]  
    st.write('**Description:**', info['description'])  
    st.write('**Characteristics:**')  
    for char in info['characteristics']:  
        st.write('- ' + char)  
    st.write('**Composition:**', info['composition'])  
    st.write('**Properties:**', info['properties'])  
    st.write('**Formation:**', info['formation'])  
  
    # Generate a text report  
    def generate_report(predicted_label, info):  
        report = 'Microstructure Report\n'  
        report += '====================\n\n'  
        report += 'Predicted Microstructure: ' + predicted_label + '\n\n'  
        report += 'Description: ' + info['description'] + '\n\n'  
        report += 'Characteristics:\n'  
        for char in info['characteristics']:  
            report += '- ' + char + '\n'  
        report += '\nComposition: ' + info['composition'] + '\n'  
        report += 'Properties: ' + info['properties'] + '\n'  
        report += 'Formation: ' + info['formation'] + '\n'  
        return report  
  
    report_content = generate_report(predicted_label, info)  
    report_bytes = io.BytesIO(report_content.encode('utf-8'))  
  
    # Download button for the report  
    st.download_button(  
        label="Download Microstructure Report",  
        data=report_bytes,  
        file_name="microstructure_report.txt",  
        mime="text/plain"  
    )  
else:  
    st.info('Please upload an image to get started.')  
