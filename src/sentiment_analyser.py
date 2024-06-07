"""This module contains the Sentiment Analysier for our Emotion Evaluator application."""

import os
import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from transformers import pipeline

from src.custom_types import SentimentType

load_dotenv()

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH")
PREDICTED_SENTIMENT_COLUMN_NAME = os.getenv("PREDICTED_SENTIMENT_COLUMN_NAME")


class SentimentAnalyser:
    """This class analyses the sentiment of a given text"""

    def __init__(
        self,
        model: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    ):
        logger.info("Initializing the sentiment analyser...")

        self.model = model
        self.pipeline = pipeline("sentiment-analysis", model=self.model)

        logger.info("Sentiment analyser initialized successfully")

    def set_model(self, model: str):
        """Sets the model to be used for sentiment analysis.

        Args:
            model (str): The model to be used.
        """
        logger.info(f"Setting the model to: {model}...")

        self.model = model
        self.pipeline = pipeline("sentiment-analysis", model=self.model)

        logger.info(f"Model set successfully")

    def get_sentiment(self, text: str) -> SentimentType:
        """Analyses the sentiment of the given text.
        Args:
            text (str): The text to preprocess.

        Returns:
            str: The sentiment of the text (either positive or negative).
        """
        logger.debug(f"Analysing sentiment of the text...")

        # Let the model generate a label for the sentiment
        sentiment = self.pipeline(text)

        # Convert to the custom SentimentType
        if sentiment[0]["label"] == "POSITIVE":
            sentiment = SentimentType.POSITIVE.value
        elif sentiment[0]["label"] == "NEGATIVE":
            sentiment = SentimentType.NEGATIVE.value
        elif sentiment[0]["label"] == "NEUTRAL":
            sentiment = SentimentType.NEUTRAL.value
        else:
            sentiment = SentimentType.NOT_UNDERSTOOD.value

        logger.debug(f"Sentiment analysed successfully: {sentiment}")

        return sentiment

    def analyse(
        self, data: pd.DataFrame, column_to_process: str = "review"
    ) -> pd.DataFrame:
        """Analyses the sentiment of the given dataframe. And adds a new column with the sentiment (predicted_sentiment).

        Args:
            data (pd.DataFrame): The input dataframe.
            column_to_process (str, optional): The column that must be processed. Defaults to "review".

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        logger.info(f"Analysing the sentiment of the data...")

        data[PREDICTED_SENTIMENT_COLUMN_NAME] = data[column_to_process].apply(
            self.get_sentiment
        )

        logger.info(f"Data analysed successfully")

        return data

    def benchmark(self, data: pd.DataFrame) -> str | dict:
        """Benchmarks the sentiment analyser on the given dataset.
           It excpects the dataframe to have a "sentiment" column with the ground truth,
           and a "predicted_sentiment" column with the predicted sentiment.

        Args:
            data (pd.DataFrame): The dataset to benchmark on.

        Returns:
            str | dict: The classification report of the sentiment analyser.
        """
        logger.info(f"Benchmarking the sentiment analyser...")

        # Get the classification report
        report = classification_report(
            data["sentiment"], data[PREDICTED_SENTIMENT_COLUMN_NAME]
        )

        logger.info(f"Sentiment analyser benchmarked successfully")

        return report
