import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io
import hashlib

# --- Helper functions ---

def hash_password(password):
    """Hashes the given password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def signup():
    """Handles user sign-up."""
    st.subheader('Sign Up')
    new_user = st.text_input('Choose a username', key='signup_user_input')
    new_password = st.text_input('Choose a password', type='password', key='signup_pass_input')
    
    if st.button('Sign Up', key='signup_button'):
        if 'users' not in st.session_state:
            st.session_state['users'] = {}
        
        if new_user in st.session_state['users']:
            st.error('Username already exists. Please choose another.')
        elif new_user == '' or new_password == '':
            st.error('Username and password cannot be empty.')
        else:
            st.session_state['users'][new_user] = hash_password(new_password)
            st.success('Sign up successful! Please log in.')
            # After signup, direct to login by setting the option and rerunning
            st.session_state['auth_option'] = 'Log In' 
            st.rerun()

def login():
    """Handles user login."""
    st.subheader('Log In')
    user = st.text_input('Username', key='login_user_input')
    password = st.text_input('Password', type='password', key='login_pass_input')
    
    if st.button('Log In', key='login_button'):
        if 'users' in st.session_state and user in st.session_state['users']:
            if st.session_state['users'][user] == hash_password(password):
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = user
                st.success(f'Welcome, {user}!')
                st.rerun() # Rerun to switch to the main app content
            else:
                st.error('Incorrect password.')
        else:
            st.error('User not found. Please sign up.')

def logout():
    """Handles user logout."""
    if st.button('Log Out', key='logout_button'):
        st.session_state['logged_in'] = False
        st.session_state['current_user'] = None
        st.info('You have been logged out.')
        st.rerun() # Rerun to go back to the authentication screen

@st.cache_resource
def load_my_model(model_path='my_model.keras'):
    """Loads the pre-trained Keras model and caches it."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'my_model.keras' is in the correct directory.")
        return None

def generate_report(prediction, probabilities, filename='microstructure_report.csv'):
    """Generates a CSV report of the classification results."""
    output = io.StringIO()
    df = pd.DataFrame({
        'Class': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    })
    df.loc[len(df)] = ['Predicted Class', prediction]
    df.to_csv(output, index=False)
    return output.getvalue(), filename # Return filename for download button


# Define the class indices mapping
class_indices = {0: 'Martensite or Bainite', 1: 'Pearlite', 2: 'Similar', 3: 'Spheroidized Cementite'}

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
        'formation': 'Formed by rapid cooling (quenching) from austenite',
        'applications': 'Tool steels, cutting tools, springs, automotive components'
    },
    'Pearlite': {
        'description': 'Pearlite is a lamellar structure consisting of alternating layers of ferrite and cementite.',
        'characteristics': [
            'Lamellar (layered) structure',
            'Alternating ferrite (light) and cementite (dark) layers',
            'Fingerprint-like appearance'
        ],
        'composition': 'Carbon content typically 0.02-0.8% C',
        'properties': 'Moderate hardness (20-40 HRC), good strength-ductility balance',
        'formation': 'Formed by slow cooling from austenite (eutectoid transformation)',
        'applications': 'Structural steels, rails, wires, general engineering applications'
    },
    'Similar': {
        'description': 'Mixed or transitional microstructures that may contain multiple phases.',
        'characteristics': [
            'Combination of different phases',
            'May include ferrite, pearlite, and other constituents',
            'Complex microstructural features'
        ],
        'composition': 'Variable carbon content depending on phases present',
        'properties': 'Properties depend on the specific phase combination',
        'formation': 'Various cooling rates and heat treatment conditions',
        'applications': 'Depends on specific microstructural composition'
    },
    'Spheroidized Cementite': {
        'description': 'Spheroidized cementite consists of spherical carbide particles in a ferrite matrix.',
        'characteristics': [
            'Spherical or globular cementite particles',
            'Cementite dispersed in ferrite matrix',
            'Uniform distribution of carbides'
        ],
        'composition': 'Carbon content typically 0.3-1.0% C',
        'properties': 'Good machinability, moderate hardness (15-25 HRC), high ductility',
        'formation': 'Formed by prolonged heating just below eutectoid temperature',
        'applications': 'Machining applications, cold forming, wire drawing'
    }
}

# --- Main app logic ---

st.title('🔬 Steel Microstructure Classification')
st.write("Upload an image of steel microstructure to get AI-powered classification and detailed analysis.")

# Initialize session state for authentication
if 'users' not in st.session_state:
    st.session_state['users'] = {}
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None
if 'auth_option' not in st.session_state:
    st.session_state['auth_option'] = 'Log In' # Default option

# Authentication Section
if not st.session_state['logged_in']:
    st.sidebar.title("Authentication")
    st.session_state['auth_option'] = st.sidebar.selectbox(
        'Choose an option', 
        ['Log In', 'Sign Up'], 
        key='auth_selectbox',
        index=0 if st.session_state['auth_option'] == 'Log In' else 1 # Maintain selection after rerun
    )
    
    if st.session_state['auth_option'] == 'Log In':
        login()
    else:
        signup()
    
    st.info("Please log in or sign up to use the Steel Microstructure Classifier.")
    st.stop() # Stop further execution until logged in

# --- Main application content (only accessible after login) ---
if st.session_state['logged_in']:
    st.sidebar.success(f'Welcome, {st.session_state["current_user"]}!')
    logout()

    # Load the trained model using st.cache_resource
    model = load_my_model()
    if model is None: # If model loading failed, stop
        st.stop()

    # Add sidebar with information
    st.sidebar.title("About This App")
    st.sidebar.write("This application uses a Convolutional Neural Network (CNN) to classify steel microstructures into four main categories:")
    st.sidebar.write("• Martensite or Bainite")
    st.sidebar.write("• Pearlite")    
    st.sidebar.write("• Similar (Mixed phases)")
    st.sidebar.write("• Spheroidized Cementite")

    # File uploader for image input    
    uploaded_file = st.file_uploader("Choose a microstructure image...", type=["jpg", "jpeg", "png", "tif"])    

    if uploaded_file is not None:
        # Display the uploaded image
        st.subheader("📸 Uploaded Image")
        
        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display the original image
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Original Microstructure Image", use_column_width=True)
        
        with col2:
            # Load and preprocess the image for prediction
            # Ensure the image is in RGB format if the model expects it (most CNNs do)
            test_image = Image.open(uploaded_file).convert('RGB').resize((128, 128))
            st.image(test_image, caption="Processed Image (128x128)", use_column_width=True)
        
        # Preprocess for model prediction
        test_image_array = image.img_to_array(test_image)
        test_image_array = np.expand_dims(test_image_array, axis=0) # Add batch dimension
        
        # Add a prediction button
        if st.button("🔍 Analyze Microstructure", type="primary"):
            with st.spinner("Analyzing microstructure..."):
                # Use the loaded model to make the prediction    
                result = model.predict(test_image_array)
                
                # Get prediction probabilities
                probabilities = result[0]
                predicted_class_index = np.argmax(probabilities)
                confidence = probabilities[predicted_class_index] * 100
                
                # Get the prediction using the class_indices mapping    
                prediction = class_indices[predicted_class_index]
                
                # Display results
                st.subheader("🎯 Classification Results")
                
                # Show prediction with confidence
                st.success(f"**Predicted Microstructure:** {prediction}")
                st.info(f"**Confidence:** {confidence:.1f}%")
                
                # Show all class probabilities
                st.subheader("📊 Prediction Probabilities")
                prob_data = {}
                for idx, class_name in class_indices.items():
                    prob_data[class_name] = probabilities[idx] * 100
                
                # Create a bar chart of probabilities
                st.bar_chart(prob_data)
                
                # Display detailed information about the predicted microstructure
                st.subheader("📋 Detailed Microstructure Report")
                
                info = microstructure_info[prediction]
                
                # Create expandable sections for different aspects
                with st.expander("🔬 Microstructure Description", expanded=True):
                    st.write(info['description'])
                
                with st.expander("⚙️ Key Characteristics"):
                    for char in info['characteristics']:
                        st.write(f"• {char}")
                
                with st.expander("🧪 Typical Composition"):
                    st.write(info['composition'])
                
                with st.expander("💪 Mechanical Properties"):
                    st.write(info['properties'])
                
                with st.expander("🔥 Formation Process"):
                    st.write(info['formation'])
                
                with st.expander("🏭 Industrial Applications"):
                    st.write(info['applications'])
                
                # Add recommendations based on the prediction
                st.subheader("💡 Recommendations")
                if prediction == 'Martensite or Bainite':
                    st.warning("⚠️ This microstructure indicates rapid cooling. Consider tempering if high toughness is required.")
                elif prediction == 'Pearlite':
                    st.info("ℹ️ This microstructure provides good strength-ductility balance. Suitable for many structural applications.")
                elif prediction == 'Spheroidized Cementite':
                    st.success("✅ This microstructure offers excellent machinability. Ideal for machining operations.")
                else:
                    st.info("ℹ️ Mixed microstructure detected. Further analysis may be needed for specific applications.")

                # Generate and offer download of the report
                csv_report, csv_filename = generate_report(prediction, prob_data)
                st.download_button(
                    label="Download Report as CSV",
                    data=csv_report,
                    file_name=csv_filename,
                    mime="text/csv",
                    key='download_report_button'
                )

# Add footer
st.markdown("---")
st.markdown("*Developed by Delwinde Sham-una*")
