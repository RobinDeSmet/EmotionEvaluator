"""This module contains the preprocessor for our Emotion Evaluator application."""

import os
import logging
import pandas as pd
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

        # self.stop_words = set(stopwords.words("english"))
        # self.lemmatizer = WordNetLemmatizer()
        # self.sid = SentimentIntensityAnalyzer()

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
