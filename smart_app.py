import streamlit as st
import cv2
import numpy as np
from PIL import Image  # Changed to appropriate import
import hashlib
import re
import math
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer  # Fixed class name
from password_strength import PasswordPolicy, PasswordStats  # Fixed class names
from io import BytesIO  # Fixed import from `bytesio` to `BytesIO`

# Download nltk resources (only need to run this once)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
# Use custom CSS to change the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #800080;
    }
    [data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    }
    [data-testid="stSidebar"] {
        background-color: #D8BFD8 ;
        background-size: cover;
    }
    [data-testid="stMarkdownContainer"] p {
        color: black;
    }
    [data-testid="stMarkdownContainer"] h1 {
        color: #DA70D6  ;
    }

    [data-testid="stFileUploader"] p{
        color: white  ;
    }
    [data-testid="stFileUploaderDropzone"] {
        color: #800080  ;
        background-color: #D8BFD8;
    }
    [data-testid="stBaseButton-secondary"] {
        color: #D8BFD8  ;
        background-color: #800080;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        color: #800080  ;
    }
    [data-testid="stMarkdownContainer"] p {
        color: white;
    }
    [data-testid="stTextAreaRootElement"] textarea {
        background-color: #D8BFD8;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("Smart App with Multiple Features")
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a feature:", ("Password Generator from Image", "Text Sentiment Analysis", "Image-to-Sketch Converter"))

# For Password Generator from Image
if option == "Password Generator from Image":
    st.header("Password Generator Based on Image")
    uploaded_file = st.file_uploader("Upload an image to generate password", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to bytes
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        # Create a hash of the image
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Use the first 12 characters of the hash as a password
        password = image_hash[:12]
        st.write(f"Generated password (for this image): {password}")

        password_input = st.text_input("Enter the generated password:", type="password")

        if password_input:
            if password_input == password:
                st.success("Correct password!")
            else:
                st.error("Incorrect password. Try again!")

# For Text Sentiment Analysis
elif option == "Text Sentiment Analysis":
    st.header("Text Sentiment Analysis")
    text_input = st.text_area("Enter your text here:", "")

    if text_input:
        # Perform sentiment analysis
        sid = SentimentIntensityAnalyzer()
        sentiment_score = sid.polarity_scores(text_input)
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Positive: {sentiment_score['pos']}")
        st.write(f"Neutral: {sentiment_score['neu']}")
        st.write(f"Negative: {sentiment_score['neg']}")
        st.write(f"Overall Sentiment: {'Positive' if sentiment_score['compound'] > 0 else 'Negative' if sentiment_score['compound'] < 0 else 'Neutral'}")

# For Image-to-Sketch Converter
elif option == "Image-to-Sketch Converter":
    st.header("Image-to-Sketch Converter")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        sketch_intensity = st.slider("Sketch Intensity", min_value=1, max_value=20, value=10, step=1)

        # Convert to sketch
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray_image)
        blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
        inverted_blurred = cv2.bitwise_not(blurred_image)
        sketch = cv2.divide(gray_image, inverted_blurred, scale=255.0)
        sketch = cv2.addWeighted(sketch, sketch_intensity / 10, sketch, 0, 0)

        st.image(sketch, caption="Converted Sketch", use_container_width=True)

        st.download_button(
            label="Download Sketch",
            data=cv2.imencode('.png', sketch)[1].tobytes(),
            file_name="sketch.png",
            mime="image/png"
        )