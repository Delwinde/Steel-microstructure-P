# Import necessary libraries
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io
import hashlib

# --- User Authentication Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup():
    st.subheader('Sign Up')
    new_user = st.text_input('Username', key='signup_user')
    new_password = st.text_input('Password', type='password', key='signup_pass')
    if st.button('Create Account'):
        if new_user and new_password:
            if 'users' not in st.session_state:
                st.session_state['users'] = {}
            if new_user in st.session_state['users']:
                st.warning('Username already exists!')
            else:
                st.session_state['users'][new_user] = hash_password(new_password)
                st.success('Account created! Please log in.')
        else:
            st.warning('Please enter a username and password.')

def login():
    st.subheader('Log In')
    user = st.text_input('Username', key='login_user')
    password = st.text_input('Password', type='password', key='login_pass')
    if st.button('Log In'):
        if 'users' in st.session_state and user in st.session_state['users']:
            if st.session_state['users'][user] == hash_password(password):
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = user
                st.experimental_rerun()  # Force rerun to show main app
            else:
                st.error('Incorrect password.')
        else:
            st.error('User not found. Please sign up.')

# --- Main App Authentication Check ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title('Microstructure Classification App')
    st.write('Please log in or sign up to continue.')
    
    option = st.selectbox('Choose an option', ['Log In', 'Sign Up'])
    if option == 'Log In':
        login()
    else:
        signup()
    st.stop()

# --- Main App Content (Only shown after login) ---
st.title('Microstructure Classification App')
st.write('Welcome, ' + st.session_state['current_user'] + '!')

# Add logout button
if st.button('Logout'):
    st.session_state['logged_in'] = False
    st.session_state['current_user'] = None
    st.experimental_rerun()

# Load the trained model
@st.cache_resource
def load_classification_model():
    return load_model('my_model.keras')

model = load_classification_model()

# Define the class indices mapping
class_indices = {0: 'Martensite or Bainite', 1: 'Pearlite', 2: 'Similar', 3: 'Spheroidized Cementite'}

# Define detailed information for each microstructure
microstructure_info = {
    'Martensite or Bainite': {
        'description': 'Martensite and Bainite are transformation products of austenite formed during rapid cooling.',
        'characteristics': [
            'Martensite: Needle-like or lath-like structure',
            'Bainite: Feathery or acicular structure',
            'High hardness and strength',
            'Formed during quenching or controlled cooling'
        ]
    },
    'Pearlite': {
        'description': 'Pearlite is a lamellar structure consisting of alternating layers of ferrite and cementite.',
        'characteristics': [
            'Lamellar (layered) structure',
            'Alternating ferrite and cementite layers',
            'Formed during slow cooling',
            'Good combination of strength and ductility'
        ]
    },
    'Similar': {
        'description': 'Similar microstructures that may share characteristics with multiple phases.',
        'characteristics': [
            'Mixed or transitional structures',
            'May contain multiple phases',
            'Requires further analysis for precise identification',
            'Common in complex heat treatment processes'
        ]
    },
    'Spheroidized Cementite': {
        'description': 'Spheroidized cementite consists of spherical cementite particles in a ferrite matrix.',
        'characteristics': [
            'Spherical cementite particles',
            'Ferrite matrix',
            'Excellent machinability',
            'Formed through spheroidizing heat treatment'
        ]
    }
}

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    
    # Make prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)
    pred_label = class_indices[pred_class]
    
    # Display results
    st.subheader('Prediction Results')
    st.write('**Predicted Microstructure:** ' + pred_label)
    st.write('**Confidence:** ' + str(round(confidence * 100, 2)) + '%')
    
    # Display detailed information
    st.subheader('Microstructure Information')
    st.write('**Description:** ' + microstructure_info[pred_label]['description'])
    st.write('**Characteristics:**')
    for char in microstructure_info[pred_label]['characteristics']:
        st.write('- ' + char)
    
    # Prepare detailed report
    report_content = f"""Microstructure Classification Report
User: {st.session_state['current_user']}
Date: {st.experimental_get_query_params().get('date', ['N/A'])[0] if st.experimental_get_query_params().get('date') else 'N/A'}

PREDICTION RESULTS:
Predicted Microstructure: {pred_label}
Confidence: {round(confidence * 100, 2)}%

MICROSTRUCTURE INFORMATION:
Description: {microstructure_info[pred_label]['description']}

Characteristics:
{chr(10).join('- ' + char for char in microstructure_info[pred_label]['characteristics'])}

CLASSIFICATION PROBABILITIES:
{chr(10).join(f'{class_indices[i]}: {round(prediction[0][i] * 100, 2)}%' for i in range(len(class_indices)))}

---
Report generated by Microstructure Classification App
"""
    
    # Download report button
    st.download_button(
        label="📄 Download Detailed Report",
        data=report_content,
        file_name=f"microstructure_report_{st.session_state['current_user']}.txt",
        mime="text/plain",
        help="Download a detailed report of the classification results"
    )
    
    # Show all class probabilities
    st.subheader('All Class Probabilities')
    for i, (class_name, prob) in enumerate(zip(class_indices.values(), prediction[0])):
        st.write(f'{class_name}: {round(prob * 100, 2)}%')

st.write('---')
st.write('Upload an image to get started with microstructure classification.')
