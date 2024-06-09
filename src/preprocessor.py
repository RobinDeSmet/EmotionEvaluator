"""This module contains the preprocessor for our Emotion Evaluator application."""

import os
import re
import logging
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.custom_types import SentimentType

load_dotenv()

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH"))


class Preprocessor:
    """This class preprocess the review dataset"""

    def __init__(self, sequence_length: int = MAX_SEQUENCE_LENGTH):
        logger.info("Initializing the preprocessor...")

        self.sequence_length = sequence_length

        logger.info("Preprocessor initialized successfully")

    def read_in_data(self, data_path: str = DATA_PATH) -> pd.DataFrame:
        """Extracts the data from the CSV file and loads it into a pandas DataFrame.

        Args:
            DATA_PATH (str, optional): The path to our dataset. Defaults to DATA_PATH.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        logger.info(f"Reading in the data from: {data_path}...")

        # Read in the csv file
        df = pd.read_csv(data_path)

        # Transform dataframe to the desired format:
        #   - Combine all the columns, right before the sentiment column into a single review column.
        #   - Create a sentiment column with 1 for positive and 0 for negative (No neutral in this dataset).
        reviews = []
        sentiments = []
        for _, row in df.iterrows():
            review = ""
            for col in df.columns:
                if row[col] != "positive" and row[col] != "negative":
                    review += f",{str(row[col])}"
                else:
                    sentiment = row[col]
                    if sentiment == "positive":
                        sentiments.append(SentimentType.POSITIVE.value)
                    elif sentiment == "negative":
                        sentiments.append(SentimentType.NEGATIVE.value)
                    elif sentiment == "neutral":
                        sentiments.append(SentimentType.NEUTRAL.value)
                    else:
                        sentiments.append(SentimentType.NOT_UNDERSTOOD.value)
                    break
            reviews.append(review[1:])

        # Create the resulting df
        resulting_df = pd.DataFrame({"review": reviews, "sentiment": sentiments})

        logger.info(f"Data loaded successfully [{len(resulting_df['review'])} entries]")
        return resulting_df

    def preprocess_text(self, text: str) -> str:
        """Apply the following preprocessing steps to the text:
            - removing HTML tags
            - removing URLs
            - removing extra whitespace
            TODO: Spelling correction

            Following things were tried but reduced the model's performance:
            - Removing special characters and numbers
            - Tokenizing the text
            - Removing stop words
            - Stemming or Lemmatizing the tokens

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        logger.info(f"Preprocessing the text: {text}")

        # Remove the HTML tags
        soup = BeautifulSoup(text, "lxml")
        text = soup.get_text()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Limit the sequence length
        logger.error(f"Sequence length: {self.sequence_length}")
        text = text[: self.sequence_length]

        logger.info(f"Text preprocessed successfully: {text}")

        # return processed_text
        return text

    def preprocess(
        self, data: pd.DataFrame, column_to_process: str = "review"
    ) -> pd.DataFrame:
        """Cleaning up and preprocessing the data.

        Args:
            data (pd.DataFrame): The input dataframe.
            column_to_process (str, optional): The column that must be processed. Defaults to "review".

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        logger.info("Preprocessing the data...")

        data[column_to_process] = data[column_to_process].apply(self.preprocess_text)

        logger.info("Data preprocessed successfully")
        return data
