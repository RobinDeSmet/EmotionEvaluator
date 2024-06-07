import streamlit as st
import requests

from src.custom_types import SentimentType

st.title("Emotion Evaluator")

st.write(
    """Welcome to Emotion Evaluator! This application evaluates the sentiment of a given text.
    """
)

text = st.text_area(
    "Enter the text you would like to evaluate:", value="", max_chars=512
)

button_clicked = st.button(
    "Evaluate",
)

if button_clicked:
    # TODO: Error handling: Put the error message in a st.error()
    # TODO: Put the base URL in .env
    response = requests.post("http://localhost:8000/evaluate/", json={"content": text})

    sentiment = response.json()["sentiment"]

    if sentiment == SentimentType.POSITIVE.value:
        sentiment = "Positive"
    elif sentiment == SentimentType.NEGATIVE.value:
        sentiment = "Negative"
    elif sentiment == SentimentType.NEUTRAL.value:
        sentiment = "Neutral"
    else:
        sentiment = "Not Understood"

    st.write(f"The sentiment of the text is: {sentiment}")
