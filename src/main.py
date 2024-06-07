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

    data = preprocessor.preprocess(data)

    logger.info(f"Data: {data.head()}")

    data = analyser.analyse(data)

    report = analyser.benchmark(data)

    logger.info(f"\n{report}")

    logger.info("Closing the app")

"""FINDINGS
    - Applying stemming or lemmatization on the input text reduces the model's performance.
      This is because the model is able to generalize well and gets extra information out
      of the form the words are in.
    - Cleaning the data however improves the model's performance.
"""

# BENCHMARK: distilbert/distilbert-base-uncased-finetuned-sst-2-english
#               precision    recall  f1-score   support

#           -1       0.85      0.91      0.88        58
#            1       0.87      0.79      0.82        42

#     accuracy                           0.86       100
#    macro avg       0.86      0.85      0.85       100
# weighted avg       0.86      0.86      0.86       100
