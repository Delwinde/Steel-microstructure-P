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

# Set up the Streamlit app layout  
st.title("🔬 Steel Microstructure Classification")  
st.write("Upload an image of steel microstructure to get AI-powered classification and detailed analysis.")  

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
        test_image = image.load_img(uploaded_file, target_size=(128, 128))
        st.image(test_image, caption="Processed Image (128x128)", use_column_width=True)
    
    # Preprocess for model prediction
    test_image_array = image.img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    
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

            # --- Downloadable Report Section ---
            import datetime
            import base64
            
            # Generate the report as a string
            report_lines = []
            report_lines.append(f"Steel Microstructure Classification Report
")
            report_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
")
            report_lines.append(f"
---
")
            report_lines.append(f"Prediction: {prediction}
")
            report_lines.append(f"Confidence: {confidence:.1f}%
")
            report_lines.append(f"
Class Probabilities:
")
            for idx, class_name in class_indices.items():
                report_lines.append(f"- {class_name}: {probabilities[idx]*100:.1f}%
")
            report_lines.append(f"
---
")
            report_lines.append(f"Microstructure Description: {info['description']}
")
            report_lines.append(f"Key Characteristics:
")
            for char in info['characteristics']:
                report_lines.append(f"  - {char}
")
            report_lines.append(f"Typical Composition: {info['composition']}
")
            report_lines.append(f"Mechanical Properties: {info['properties']}
")
            report_lines.append(f"Formation Process: {info['formation']}
")
            report_lines.append(f"Industrial Applications: {info['applications']}
")
            
            # Add recommendations
            if prediction == 'Martensite or Bainite':
                report_lines.append("Recommendation: This microstructure indicates rapid cooling. Consider tempering if high toughness is required.
")
            elif prediction == 'Pearlite':
                report_lines.append("Recommendation: This microstructure provides good strength-ductility balance. Suitable for many structural applications.
")
            elif prediction == 'Spheroidized Cementite':
                report_lines.append("Recommendation: This microstructure offers excellent machinability. Ideal for machining operations.
")
            else:
                report_lines.append("Recommendation: Mixed microstructure detected. Further analysis may be needed for specific applications.
")
            
            report_str = ''.join(report_lines)
            
            # Encode report for download
            b64 = base64.b64encode(report_str.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="microstructure_report.txt">📥 Download Full Report</a>'
            st.markdown(href, unsafe_allow_html=True)



# Add footer
st.markdown("---")
st.markdown("*Developed Delwinde Sham-una*")
