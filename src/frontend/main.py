import streamlit as st
import requests
import logging
from http import HTTPStatus
from src.custom_types import SentimentType
from src.frontend.utils import url
from src.utils import configure_logging

configure_logging()

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
    response = requests.post(url("/evaluate/"), json={"content": text})

    if response.status_code != HTTPStatus.OK:
        st.error(f"{response.json()['error']}")
    else:
        sentiment = response.json()["sentiment"]

        if sentiment == SentimentType.POSITIVE.value:
            sentiment = "Positive 👍"
        elif sentiment == SentimentType.NEGATIVE.value:
            sentiment = "Negative 👎"
        elif sentiment == SentimentType.NEUTRAL.value:
            sentiment = "Neutral 😐"
        else:
            sentiment = "Not Understood 🤔"

        st.write(f"The sentiment of the text is: {sentiment}")
