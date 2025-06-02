import streamlit as st  
import pandas as pd  
import numpy as np  
import hashlib  
import io  
  
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
            st.success('Sign up successful! Please log in.')  
            st.session_state['do_rerun'] = True  
  
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
    if st.button('Log Out'):  
        st.session_state['logged_in'] = False  
        st.session_state['current_user'] = None  
        st.session_state['do_rerun'] = True  
  
def generate_report(prediction, probabilities, filename='microstructure_report.csv'):  
    output = io.StringIO()  
    df = pd.DataFrame({  
        'Class': list(probabilities.keys()),  
        'Probability': list(probabilities.values())  
    })  
    df.loc[len(df)] = ['Predicted Class', prediction]  
    df.to_csv(output, index=False)  
    return output.getvalue()  
  
# --- Main app logic ---  
  
st.title('Steel Microstructure Classification')  
  
# Initialize session state  
if 'users' not in st.session_state:  
    st.session_state['users'] = {}  
if 'logged_in' not in st.session_state:  
    st.session_state['logged_in'] = False  
if 'current_user' not in st.session_state:  
    st.session_state['current_user'] = None  
  
# Authentication  
if not st.session_state['logged_in']:  
    option = st.selectbox('Choose an option', ['Log In', 'Sign Up'])  
    if option == 'Log In':  
        login()  
    else:  
        signup()  
    # Handle rerun after login/signup  
    if st.session_state.get('do_rerun', False):  
        st.session_state['do_rerun'] = False  
        st.experimental_rerun()  
    st.stop()  
  
# Main app after login  
st.success('Welcome, ' + st.session_state['current_user'] + '!')  
logout()  
  
st.header('Upload Microstructure Data')  
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])  
  
if uploaded_file is not None:  
    df = pd.read_csv(uploaded_file)  
    st.write('Preview of uploaded data:')  
    st.dataframe(df.head())  
  
    # Simulate prediction (replace with your model)  
    st.header('Classification Result')  
    classes = ['Ferrite', 'Pearlite', 'Martensite', 'Austenite']  
    probabilities = dict(zip(classes, np.random.dirichlet(np.ones(len(classes)), size=1)[0]))  
    prediction = max(probabilities, key=probabilities.get)  
    st.write('**Predicted Class:**', prediction)  
    st.write('**Class Probabilities:**')  
    st.dataframe(pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability']))  
  
    # Download report  
    report_csv = generate_report(prediction, probabilities)  
    st.download_button(  
        label='Download Detailed Report',  
        data=report_csv,  
        file_name='microstructure_report.csv',  
        mime='text/csv'  
    )  
else:  
    st.info('Please upload a CSV file to begin.')  
  
