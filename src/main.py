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

    logger.info("Closing the app")

# RESULTS

# Nltk VADER model:
#  precision    recall  f1-score   support

#           -1       0.74      0.45      0.56        58
#            1       0.51      0.79      0.62        42

#     accuracy                           0.59       100
#    macro avg       0.63      0.62      0.59       100
# weighted avg       0.64      0.59      0.58       100
