import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io
import hashlib
import base64
from weasyprint import HTML # Import WeasyPrint

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
                st.rerun()
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
        st.rerun()

@st.cache_resource
def load_my_model(model_path='my_model.keras'):
    """Loads the pre-trained Keras model and caches it."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'my_model.keras' is in the correct directory.")
        return None

def generate_html_content(prediction, confidence, prob_data, info, image_base64=None):
    """Generates the HTML content for the report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Steel Microstructure Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 14px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            h1 {{ font-size: 2em; text-align: center; margin-bottom: 20px; }}
            h2 {{ font-size: 1.5em; margin-top: 20px; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
            h3 {{ font-size: 1.2em; margin-top: 15px; color: #0056b3; }}
            .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 8px; }}
            .section-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; color: #0056b3; }}
            .result-box {{ background-color: #e6ffe6; border-left: 5px solid #4CAF50; padding: 10px; margin-bottom: 15px; }}
            .info-box {{ background-color: #e0f2f7; border-left: 5px solid #2196F3; padding: 10px; margin-bottom: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            ul {{ list-style-type: disc; margin-left: 20px; }}
            .image-container {{ text-align: center; margin-top: 20px; }}
            .image-container img {{ max-width: 500px; height: auto; border: 1px solid #ddd; padding: 5px; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Steel Microstructure Analysis Report</h1>
        <p>Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section result-box">
            <div class="section-title">Classification Results</div>
            <p><strong>Predicted Microstructure:</strong> {prediction}</p>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
    """
    
    if image_base64:
        html_content += f"""
        <div class="section image-container">
            <div class="section-title">Uploaded Microstructure Image</div>
            <img src="data:image/png;base64,{image_base64}" alt="Uploaded Microstructure">
        </div>
        """

    html_content += """
        <div class="section info-box">
            <div class="section-title">Prediction Probabilities</div>
            <table>
                <tr><th>Class</th><th>Probability (%)</th></tr>
    """
    for class_name, prob in prob_data.items():
        html_content += f"<tr><td>{class_name}</td><td>{prob:.1f}</td></tr>"

    html_content += f"""
            </table>
        </div>

        <div class="section">
            <div class="section-title">Detailed Microstructure Information: {prediction}</div>
            <h3>Description</h3>
            <p>{info['description']}</p>
            
            <h3>Key Characteristics</h3>
            <ul>
    """
    for char in info['characteristics']:
        html_content += f"<li>{char}</li>"
    
    html_content += f"""
            </ul>
            
            <h3>Typical Composition</h3>
            <p>{info['composition']}</p>
            
            <h3>Mechanical Properties</h3>
            <p>{info['properties']}</p>
            
            <h3>Formation Process</h3>
            <p>{info['formation']}</p>
            
            <h3>Industrial Applications</h3>
            <p>{info['applications']}</p>
        </div>

        <div class="section result-box">
            <div class="section-title">Recommendations</div>
    """
    if prediction == 'Martensite or Bainite':
        html_content += "<p>⚠️ This microstructure indicates rapid cooling. Consider tempering if high toughness is required.</p>"
    elif prediction == 'Pearlite':
        html_content += "<p>ℹ️ This microstructure provides good strength-ductility balance. Suitable for many structural applications.</p>"
    elif prediction == 'Spheroidized Cementite':
        html_content += "<p>✅ This microstructure offers excellent machinability. Ideal for machining operations.</p>"
    else:
        html_content += "<p>ℹ️ Mixed microstructure detected. Further analysis may be needed for specific applications.</p>"

    html_content += """
        </div>

        <p class="footer">Developed by Delwinde Sham-una</p>
    </body>
    </html>
    """
    return html_content

def convert_html_to_pdf(html_string):
    """Converts an HTML string to a PDF byte stream."""
    try:
        # WeasyPrint directly reads from a string
        pdf_bytes = HTML(string=html_string).write_pdf()
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF: {e}. Ensure WeasyPrint and its system dependencies are correctly installed.")
        return None

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
    st.session_state['auth_option'] = 'Log In'

# Authentication Section
if not st.session_state['logged_in']:
    st.sidebar.title("Authentication")
    st.session_state['auth_option'] = st.sidebar.selectbox(
        'Choose an option', 
        ['Log In', 'Sign Up'], 
        key='auth_selectbox',
        index=0 if st.session_state['auth_option'] == 'Log In' else 1
    )
    
    if st.session_state['auth_option'] == 'Log In':
        login()
    else:
        signup()
    
    st.info("Please log in or sign up to use the Steel Microstructure Classifier.")
    st.stop()

# --- Main application content (only accessible after login) ---
if st.session_state['logged_in']:
    st.sidebar.success(f'Welcome, {st.session_state["current_user"]}!')
    logout()

    # Load the trained model using st.cache_resource
    model = load_my_model()
    if model is None:
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
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Original Microstructure Image", use_column_width=True)
            
            # Prepare image for embedding in HTML report
            img_byte_arr = io.BytesIO()
            display_image.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        with col2:
            test_image = Image.open(uploaded_file).convert('RGB').resize((128, 128))
            st.image(test_image, caption="Processed Image (128x128)", use_column_width=True)
        
        test_image_array = image.img_to_array(test_image)
        test_image_array = np.expand_dims(test_image_array, axis=0)
        
        if st.button("🔍 Analyze Microstructure", type="primary"):
            with st.spinner("Analyzing microstructure..."):
                result = model.predict(test_image_array)
                
                probabilities = result[0]
                predicted_class_index = np.argmax(probabilities)
                confidence = probabilities[predicted_class_index] * 100
                
                prediction = class_indices[predicted_class_index]
                
                st.subheader("🎯 Classification Results")
                st.success(f"**Predicted Microstructure:** {prediction}")
                st.info(f"**Confidence:** {confidence:.1f}%")
                
                st.subheader("📊 Prediction Probabilities")
                prob_data = {}
                for idx, class_name in class_indices.items():
                    prob_data[class_name] = probabilities[idx] * 100
                
                st.bar_chart(prob_data)
                
                st.subheader("📋 Detailed Microstructure Report")
                
                info = microstructure_info[prediction]
                
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
                
                st.subheader("💡 Recommendations")
                if prediction == 'Martensite or Bainite':
                    st.warning("⚠️ This microstructure indicates rapid cooling. Consider tempering if high toughness is required.")
                elif prediction == 'Pearlite':
                    st.info("ℹ️ This microstructure provides good strength-ductility balance. Suitable for many structural applications.")
                elif prediction == 'Spheroidized Cementite':
                    st.success("✅ This microstructure offers excellent machinability. Ideal for machining operations.")
                else:
                    st.info("ℹ️ Mixed microstructure detected. Further analysis may be needed for specific applications.")

                # Generate HTML content
                html_report_content = generate_html_content(
                    prediction, confidence, prob_data, info, image_base64=img_base64
                )
                
                # Convert HTML content to PDF bytes
                pdf_report_bytes = convert_html_to_pdf(html_report_content)

                if pdf_report_bytes:
                    st.download_button(
                        label="Download Full Report (PDF)",
                        data=pdf_report_bytes,
                        file_name="microstructure_report.pdf",
                        mime="application/pdf",
                        key='download_pdf_report_button'
                    )
                else:
                    st.error("Could not generate PDF report. Check logs for details.")
                
# Add footer
st.markdown("---")
st.markdown("*Developed by Delwinde Sham-una*")
