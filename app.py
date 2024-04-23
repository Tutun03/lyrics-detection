# Import necessary modules
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignore warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load pre-trained models and vectorizer
lg = pickle.load(open('logistic_regression.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# Function to clean text
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)  # Only keep letters
    text = text.lower()  # Lowercase
    text = text.split()  # Split into words
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Function to predict emotion
def predict_lyrics(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    
    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    
    return predicted_emotion

# Set up the Streamlit app
st.set_page_config(page_title="Lyrics Emotion Detector", page_icon="🎵", layout="centered")

# CSS styling with gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f5f7fa, #c3cfe2); /* Gradient background */
        padding: 20px;
    }
    .stTextInput, .stButton {
        border-radius: 10px; /* Rounded edges */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title(" Emotion Detection System")
st.subheader("Enter song lyrics to predict the emotion.")

# User input
user_input = st.text_area("Enter lyrics here:")

# Predict button and output
if st.button("Predict"):
    predicted_emotion = predict_lyrics(user_input)
    st.write("Predicted Emotion:", predicted_emotion)
else:
    st.write("Click 'Predict' to see the emotion prediction.")
