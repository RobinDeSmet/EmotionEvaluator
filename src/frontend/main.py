"""This module contains the Streamlit frontend for the Emotion Evaluator application."""

import os
from http import HTTPStatus

import streamlit as st
import requests
from dotenv import load_dotenv

from src.custom_types import SentimentType
from src.frontend.utils import get_url
from src.utils import configure_logging

configure_logging()

load_dotenv()

INPUT_TEXT_MAX_LENGTH = int(os.getenv("INPUT_TEXT_MAX_LENGTH"))

st.title("Emotion Evaluator")

st.write(
    """Welcome to Emotion Evaluator! This application evaluates the sentiment of a given text.
    """
)

text = st.text_area(
    "Enter the text you would like to evaluate:",
    value="",
    max_chars=INPUT_TEXT_MAX_LENGTH,
)

button_clicked = st.button(
    "Evaluate",
)

if button_clicked:
    response = requests.post(get_url("/evaluate/"), json={"content": text}, timeout=5)

    if response.status_code != HTTPStatus.OK:
        st.error(f"{response.json()['error']}")
    else:
        SENTIMENT = response.json()["sentiment"]

        if SENTIMENT == SentimentType.POSITIVE.value:
            SENTIMENT = "Positive 👍"
        elif SENTIMENT == SentimentType.NEGATIVE.value:
            SENTIMENT = "Negative 👎"
        elif SENTIMENT == SentimentType.NEUTRAL.value:
            SENTIMENT = "Neutral 😐"
        else:
            SENTIMENT = "Not Understood 🤔"

        st.write(f"The sentiment of the text is: {SENTIMENT}")
