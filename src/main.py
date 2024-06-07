"""Main entrypoint for the application"""

import logging
import sys
import pandas as pd
from src.utils import configure_logging
from src.preprocessor import Preprocessor

if __name__ == "__main__":

    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting the app...")
    logger.info("Arguments: %s", sys.argv)

    preprocessor = Preprocessor()

    data = preprocessor.read_in_data()

    logger.info(data.head())

    data = preprocessor.preprocess(data)

    logger.info(data.head())

    logger.info("Closing the app")
