import streamlit as st  
import numpy as np  
import pandas as pd  
import hashlib  
from PIL import Image  
import io  
import os  
  
# --- Helper functions ---  
  
def hash_password(password):  
    return hashlib.sha256(password.encode()).hexdigest()  
  
def signup():  
    st.subheader('Sign Up')  
    new_user = st.text_input('Choose a username', key='signup_user')  
    new_password = st.text_input('Choose a password', type='password', key='signup_pass')  
    if st.button('Sign Up'):  
        if 'users' not in st.session_state:  
            st.session_state['users'] = {}  
        if new_user in st.session_state['users']:  
            st.error('Username already exists. Please choose another.')  
        elif new_user == '' or new_password == '':  
            st.error('Username and password cannot be empty.')  
        else:  
            st.session_state['users'][new_user] = hash_password(new_password)  
            st.success('Account created! Please log in.')  
  
def login():  
    st.subheader('Log In')  
    user = st.text_input('Username', key='login_user')  
    password = st.text_input('Password', type='password', key='login_pass')  
    if st.button('Log In'):  
        if 'users' in st.session_state and user in st.session_state['users']:  
            if st.session_state['users'][user] == hash_password(password):  
                st.session_state['logged_in'] = True  
                st.session_state['current_user'] = user  
                st.session_state['do_rerun'] = True  
            else:  
                st.error('Incorrect password.')  
        else:  
            st.error('User not found. Please sign up.')  
  
def logout():  
    if st.button('Logout'):  
        st.session_state['logged_in'] = False  
        st.session_state['current_user'] = None  
        st.session_state['do_rerun'] = True  
  
def simulate_classification(image_name):  
    # Simulate class probabilities  
    classes = ['Pearlite', 'Bainite', 'Martensite', 'Austenite']  
    probabilities = np.random.dirichlet(np.ones(len(classes)), size=1)[0]  
    prediction = classes[np.argmax(probabilities)]  
    return prediction, dict(zip(classes, probabilities))  
  
def generate_report(results):  
    # results: list of dicts with keys: image, prediction, probabilities  
    rows = []  
    for res in results:  
        row = {'Image': res['image'], 'Predicted Class': res['prediction']}  
        row.update(res['probabilities'])  
        rows.append(row)  
    df = pd.DataFrame(rows)  
    return df.to_csv(index=False).encode('utf-8')  
  
# --- Main App ---  
  
st.title('Steel Microstructure Classification')  
  
if 'logged_in' not in st.session_state:  
    st.session_state['logged_in'] = False  
  
if not st.session_state.get('logged_in', False):  
    option = st.selectbox('Choose an option', ['Log In', 'Sign Up'])  
    if option == 'Log In':  
        login()  
    else:  
        signup()  
    if st.session_state.get('do_rerun', False):  
        st.session_state['do_rerun'] = False  
        st.experimental_rerun()  
    st.stop()  
  
st.success(f"Welcome, {st.session_state.get('current_user', 'User')}!")  
logout()  
if st.session_state.get('do_rerun', False):  
    st.session_state['do_rerun'] = False  
    st.experimental_rerun()  
  
st.header('Upload Microstructure Images')  
uploaded_files = st.file_uploader(  
    "Upload PNG, JPG, or TIF images",   
    type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],   
    accept_multiple_files=True  
)  
  
if uploaded_files:  
    results = []  
    for uploaded_file in uploaded_files:  
        st.subheader(f"Image: {uploaded_file.name}")  
        image = Image.open(uploaded_file)  
        st.image(image, caption=uploaded_file.name, use_column_width=True)  
        prediction, probabilities = simulate_classification(uploaded_file.name)  
        st.write('**Predicted Class:**', prediction)  
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])  
        st.dataframe(prob_df)  
        results.append({  
            'image': uploaded_file.name,  
            'prediction': prediction,  
            'probabilities': probabilities  
        })  
    # Download report for all images  
    report_csv = generate_report(results)  
    st.download_button(  
        label='Download Detailed Report',  
        data=report_csv,  
        file_name='microstructure_report.csv',  
        mime='text/csv'  
    )  
else:  
    st.info('Please upload one or more image files to begin.')  
