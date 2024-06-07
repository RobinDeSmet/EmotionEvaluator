"""This module contains the preprocessor for our Emotion Evaluator application."""

import os
import logging
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH")


class Preprocessor:
    """This class preprocess the review dataset"""

    def __init__(self):
        logger.info("Initializing the preprocessor...")

        self.word_tokenizer = word_tokenize
        self.stop_words = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()

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
        #   - Create a sentiment column with 1 for positive and 0 for negative (No neutral).
        reviews = []
        sentiments = []
        for _, row in df.iterrows():
            review = ""
            for col in df.columns:
                if row[col] != "positive" and row[col] != "negative":
                    review += f",{str(row[col])}"
                else:
                    sentiments.append(1 if row[col] == "positive" else 0)
                    break
            reviews.append(review[1:])

        # Create the resulting df
        resulting_df = pd.DataFrame({"review": reviews, "sentiment": sentiments})

        logger.info(f"Data loaded successfully [{len(resulting_df['review'])} entries]")
        return resulting_df

    def preprocess_text(self, text: str) -> str:
        """Apply the following preprocessing steps to the text:
            - removing HTML tags
            - tokenizing
            - removing stop words
            - lemmatizing the text.
        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        logger.debug(f"Preprocessing the text: {text}...")

        # Remove the HTML tags
        logger.debug(f"Removing HTML tags...")

        soup = BeautifulSoup(text, "lxml")
        text = soup.get_text()

        # Tokenize the text
        logger.debug(f"Tokenizing the text...")

        tokens = self.word_tokenizer(text.lower())

        # Remove stop words
        logger.debug(f"Removing stop words...")

        filtered_tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize the tokens
        logger.debug(f"Lemmatizing the tokens...")
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token) for token in filtered_tokens
        ]

        # Join the tokens back into a string
        processed_text = " ".join(lemmatized_tokens)

        logger.debug(f"Text preprocessed successfully: {processed_text}")
        return processed_text

    def preprocess(
        self, data: pd.DataFrame, column_to_process: str = "review"
    ) -> pd.DataFrame:
        """Preprocess the data by:
            - removing HTML tags
            - tokenizing
            - removing stop words
            - lemmatizing the text.

        Args:
            data (pd.DataFrame): The input dataframe.
            column_to_process (str, optional): The column that must be processed. Defaults to "review".

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        logger.info(f"Preprocessing the data...")

        data[column_to_process] = data[column_to_process].apply(self.preprocess_text)

        logger.info(f"Data preprocessed successfully")
        return data
