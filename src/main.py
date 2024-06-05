"""Main entrypoint for the application"""

import logging
import sys
from src.utils import configure_logging
from src.extract_data import extract_data

if __name__ == "__main__":

    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting the app...")
    logger.info("Arguments: %s", sys.argv)

    df = extract_data()
    logger.info(f"Data loaded successfully: {df.head()}")
    logger.info("Closing the app")
