"""Main entrypoint for the application"""

import logging
import sys
import pandas as pd
from src.utils import configure_logging
from src.preprocessor import Preprocessor
from src.sentiment_analyser import SentimentAnalyser

if __name__ == "__main__":

    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting the app...")
    logger.info("Arguments: %s", sys.argv)

    preprocessor = Preprocessor()
    analyser = SentimentAnalyser()

    data = preprocessor.read_in_data()

    logger.info(data.head())

    data = preprocessor.preprocess(data)

    logger.info(data.head())

    analyser.get_sentiment(data["review"].iloc[0])

    analyser.get_sentiment(data["review"].iloc[1])

    data = analyser.analyse(data)

    report = analyser.benchmark(data)

    logger.info(f"\n{report}")
    print(f"{data['sentiment']}")
    print(f"{data['predicted_sentiment']}")

    logger.info("Closing the app")
