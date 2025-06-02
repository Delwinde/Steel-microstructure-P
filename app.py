# Add user authentication (sign up and log in) using Streamlit session state
# This is a code snippet to be integrated at the top of your Streamlit app
import streamlit as st
import hashlib

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

# Show login/signup or app based on session state
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
    # The rest of your app goes here
