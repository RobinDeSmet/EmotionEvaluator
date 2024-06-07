"""This module contains the Sentiment Analysier for our Emotion Evaluator application."""

import os
import logging
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from sklearn.metrics import classification_report

from src.custom_types import SentimentType

load_dotenv()

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH")
PREDICTED_SENTIMENT_COLUMN_NAME = os.getenv("PREDICTED_SENTIMENT_COLUMN_NAME")


class SentimentAnalyser:
    """This class analyses the sentiment of a given text"""

    def __init__(self, model=SentimentIntensityAnalyzer()):
        logger.info("Initializing the sentiment analyser...")

        self.model = model

        logger.info("Sentiment analyser initialized successfully")

    def get_sentiment(self, text: str) -> SentimentType:
        """Analyses the sentiment of the given text.
        Args:
            text (str): The text to preprocess.

        Returns:
            str: The sentiment of the text (either positive or negative).
        """
        logger.debug(f"Analysing sentiment of the text...")

        if isinstance(self.model, SentimentIntensityAnalyzer):
            # Get the polarity scores of the text
            scores = self.model.polarity_scores(text)

            logger.debug(scores)

            # Determine the sentiment
            sentiment = (
                SentimentType.POSITIVE.value
                if scores["compound"] > 0
                else SentimentType.NEGATIVE.value
            )

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
