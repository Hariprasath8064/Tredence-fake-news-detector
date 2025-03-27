import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load model
model = tf.keras.models.load_model("model.h5")

# Define max length (should match training settings)
max_len = 100

# Prediction function
def predict_news(news_text, threshold=0.9):
    seq = tokenizer.texts_to_sequences([news_text])
    padded_seq = pad_sequences(seq, maxlen=max_len)
    
    prediction = model.predict(padded_seq)[0][0]
    confidence = round(prediction * 100, 2) if prediction > threshold else round((1 - prediction) * 100, 2)
    label = "Fake" if prediction > threshold else "Real"
    
    return label, confidence

# Predefined examples
example_news_list = [
    # Fake News Examples
    "Did they post their votes for Hillary already?",
    "VP Joe Biden: Yeah, I’m going to run in 2020.",
    "ESPN’s Wilbon on Kaepernick: ’This Is a  - There’s No Question About It’ - Breitbart",
    
    # Real News Examples
    "The Atlantic said that The NRA has a new favorite toy, but there are no bullets involved.",
    "U.S. senator: Cuba ambassador will not be approved this year",
    "Travel Ban enhances Persian New Year Celebration - The New York Times"
]

# Streamlit UI
st.title("Fake News & Deepfake Detection")

# News Headline Input
st.subheader("Check a News Headline")
news_input = st.text_input("Enter a headline to check if it's real or fake")

if st.button("Check News"):
    if news_input:
        label, confidence = predict_news(news_input)
        st.write(f"### Result: {label} (Confidence: {confidence}%)")
    else:
        st.warning("Please enter a headline.")

# Predefined Examples Section
st.subheader("Try Some Example Headlines")

# Display examples in two sections: Fake and Real
col1, col2 = st.columns(2)

with col1:
    st.write("### Fake News Examples")
    for example in example_news_list[:3]:
        if st.button(example, key=example):
            label, confidence = predict_news(example)
            st.write(f"**Result:** {label} (Confidence: {confidence}%)")

with col2:
    st.write("### Real News Examples")
    for example in example_news_list[3:]:
        if st.button(example, key=example):
            label, confidence = predict_news(example)
            st.write(f"**Result:** {label} (Confidence: {confidence}%)")

# Video Upload Section (Placeholder for Deepfake Detection)
st.subheader("Upload a Video for Deepfake Detection")
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.write("Deepfake detection model integration pending...")
    # TODO: Add deepfake detection logic

st.write("📌 **Note:** The deepfake detection feature is under development.")