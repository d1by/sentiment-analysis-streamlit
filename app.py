import streamlit as st
import joblib
from utils import preprocessor

st.title("Sentiment Analyzer")

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.joblib")

model = load_model()

user_input = st.text_area("Enter review:")

if user_input:
    prediction = model.predict([user_input])[0]
    st.success(f"Predicted Sentiment: **{prediction}**")