# Add login and signup functionality to the Streamlit app
# Read the current enhanced app code
with open('enhanced_app_with_download.py', 'r') as f:
    app_code = f.read()

# Create the authentication system code to insert at the beginning
auth_code = '''import streamlit as st
import hashlib
import json
import os
from datetime import datetime

# Authentication functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)

def authenticate_user(username, password):
    """Authenticate user credentials"""
    users = load_users()
    if username in users:
        return users[username]['password'] == hash_password(password)
    return False

def register_user(username, password, email):
    """Register new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.now().isoformat()
    }
    save_users(users)
    return True, "User registered successfully"

def login_page():
    """Display login page"""
    st.title("🔐 Steel Microstructure Classifier - Login")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create New Account")
        new_username = st.text_input("Choose Username", key="signup_username")
        new_email = st.text_input("Email Address", key="signup_email")
        new_password = st.text_input("Choose Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up", key="signup_btn"):
            if not new_username or not new_email or not new_password:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                success, message = register_user(new_username, new_password, new_email)
                if success:
                    st.success(message)
                    st.info("Please login with your new credentials")
                else:
                    st.error(message)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Check if user is logged in
if not st.session_state.logged_in:
    login_page()
    st.stop()

# Add logout button in sidebar
with st.sidebar:
    st.write(f"Welcome, {st.session_state.username}!")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

'''

# Find where to insert the auth code (after imports but before the main app)
import_end = app_code.find('st.set_page_config')
if import_end == -1:
    import_end = app_code.find('import streamlit as st') + len('import streamlit as st\
')

# Insert the authentication code
new_app_code = app_code[:import_end] + '\
' + auth_code + '\
' + app_code[import_end:]

# Save the new app with authentication
with open('app_with_auth.py', 'w') as f:
    f.write(new_app_code)

print("Authentication system added successfully!")
print("Features included:")
print("- User registration with username, email, and password")
print("- Secure password hashing using SHA-256")
print("- Login/logout functionality")
print("- Session management")
print("- User data stored in users.json file")
print("- Welcome message and logout button in sidebar")
