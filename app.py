
import streamlit as st
import os
from PIL import Image
import plotly.graph_objects as go
from model import SteelClassifier

st.set_page_config(
    page_title="Steel Microstructure Classifier",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    try:
        model_path = 'classfier_1.h5'
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
            return None
        return SteelClassifier(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_probability_chart(probabilities):
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=values,
            marker_color='rgb(26, 118, 255)'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Classification Probabilities",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Microstructure Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template='plotly_white'
    )
    
    return fig

def main():
    st.title("🔬 Steel Microstructure Classification")
    st.write("Upload an image of steel microstructure for automatic classification")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("ℹ️ Supported image formats: JPG, JPEG, PNG")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Initialize model
    with st.spinner("🔄 Initializing model..."):
        classifier = load_classifier()
    
    if classifier is None:
        st.error("❌ Model initialization failed. Please check if the model file exists and is valid.")
        st.stop()
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner("🔄 Analyzing microstructure..."):
                result = classifier.predict(image)
            
            # Display results in the second column
            with col2:
                st.success(f"🎯 Predicted Class: {result['class']}")
                st.info(f"📊 Confidence: {result['confidence']:.2%}")
                
                # Display probability chart
                st.plotly_chart(create_probability_chart(result['probabilities']), use_container_width=True)
                
                # Detailed probabilities
                st.subheader("Detailed Analysis")
                for class_name, prob in result['probabilities'].items():
                    st.text(f"▫️ {class_name}: {prob:.2%}")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.write("Please try uploading a different image or check if the image format is supported.")

if __name__ == "__main__":
    main()
