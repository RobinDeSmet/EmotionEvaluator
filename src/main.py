"""Main entrypoint for the application"""

import logging
import sys
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
