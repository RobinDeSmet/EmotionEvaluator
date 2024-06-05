"""This module extracts the data from the IMDB review dataset and loads it into a pandas DataFrame."""

import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

load_dotenv()

DATA_LOCATION = os.getenv("DATA_LOCATION")


def extract_data() -> pd.DataFrame:
    """Extracts the data from the IMBD review dataset and loads it into a pandas DataFrame.

    Returns:
        pd.DataFrame: The IMDB review dataset as a pandas DataFrame.
    """
    # Read in the csv file
    df = pd.read_csv(DATA_LOCATION)

    # Extract the relevant columns
    df = df[["review", "sentiment"]]

    # TODO: The review is spread over several columns, we need to concatenate them
    return df
