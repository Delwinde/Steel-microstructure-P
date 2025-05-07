
import streamlit as st
from PIL import Image
import io
from model import SteelClassifier
import plotly.graph_objects as go

# Initialize the model
@st.cache_resource
def load_classifier():
    return SteelClassifier()

def create_probability_chart(probabilities):
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(x=classes, y=values)
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Steel Microstructure Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1]
    )
    
    return fig

def main():
    st.title("Steel Microstructure Classification")
    st.write("Upload an image of steel microstructure for classification")
    
    # Initialize model
    classifier = load_classifier()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            result = classifier.predict(image)
        
        # Display results
        st.success(f"Predicted Class: {result['class']}")
        st.info(f"Confidence: {result['confidence']:.2%}")
        
        # Display probability chart
        st.plotly_chart(create_probability_chart(result['probabilities']))
        
        # Additional information
        st.subheader("All Probabilities")
        for class_name, prob in result['probabilities'].items():
            st.text(f"{class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()
