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
                st.success('Logged in as ' + user)  
            else:  
                st.error('Incorrect password.')  
        else:  
            st.error('User not found. Please sign up.')  
  
# --- Authentication Gate ---  
if 'logged_in' not in st.session_state:  
    st.session_state['logged_in'] = False  
  
if not st.session_state['logged_in']:  
    option = st.selectbox('Choose an option', ['Log In', 'Sign Up'])  
    if option == 'Log In':  
        login()  
    else:  
        signup()  
    st.stop()  
else:  
    st.write('Welcome, ' + st.session_state['current_user'] + '!')  
  
# --- Load the trained model ---  
model = load_model('my_model.keras')  
  
# --- Define the class indices mapping ---  
class_indices = {0: 'Martensite or Bainite', 1: 'Pearlite', 2: 'Similar', 3: 'Spheroidized Cementite'}  
  
# --- Define detailed information for each microstructure ---  
microstructure_info = {  
    'Martensite or Bainite': {  
        'description': 'Martensite and Bainite are transformation products of austenite formed during rapid cooling.',  
        'characteristics': [  
            'Martensite: Needle-like or lath-like structure',  
            'Bainite: Fine, feathery structure'  
        ]  
    },  
    'Pearlite': {  
        'description': 'Pearlite is a two-phased, lamellar (or layered) structure composed of alternating layers of ferrite and cementite.',  
        'characteristics': [  
            'Lamellar structure',  
            'Alternating ferrite and cementite'  
        ]  
    },  
    'Similar': {  
        'description': 'Microstructure is similar to the reference image.',  
        'characteristics': [  
            'Visual similarity to reference'  
        ]  
    },  
    'Spheroidized Cementite': {  
        'description': 'Spheroidized cementite consists of rounded cementite particles in a ferrite matrix.',  
        'characteristics': [  
            'Rounded cementite particles',  
            'Ferrite matrix'  
        ]  
    }  
}  
  
# --- Main App Functionality ---  
st.title('Microstructure Classifier')  
  
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  
report_text = ""  
  
if uploaded_file is not None:  
    img = Image.open(uploaded_file)  
    st.image(img, caption='Uploaded Image', use_column_width=True)  
    img = img.resize((224, 224))  
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0)  
    x = x / 255.0  
  
    preds = model.predict(x)  
    pred_class = np.argmax(preds, axis=1)[0]  
    pred_label = class_indices[pred_class]  
  
    st.subheader('Prediction')  
    st.write('Predicted Microstructure: **' + pred_label + '**')  
    st.write('Description:', microstructure_info[pred_label]['description'])  
    st.write('Characteristics:')  
    for char in microstructure_info[pred_label]['characteristics']:  
        st.write('- ' + char)  
  
    # Prepare report text  
    report_text = (  
        "User: " + st.session_state['current_user'] + "\n"  
        "Predicted Microstructure: " + pred_label + "\n"  
        "Description: " + microstructure_info[pred_label]['description'] + "\n"  
        "Characteristics:\n" + "\n".join("- " + c for c in microstructure_info[pred_label]['characteristics'])  
    )  
  
    # Download report button  
    st.download_button(  
        label="Download Report",  
        data=report_text,  
        file_name="microstructure_report.txt",  
        mime="text/plain"  
    )  
