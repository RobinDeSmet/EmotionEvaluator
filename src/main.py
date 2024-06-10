"""Main entrypoint for the application"""

import argparse
import logging
import os

from dotenv import load_dotenv
from src.utils import configure_logging
from src.preprocessor import Preprocessor
from src.sentiment_analyser import SentimentAnalyser

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
MODEL = os.getenv("MODEL")

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI tool.")
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        type=str,
        help="Path to the CSV file that holds the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="src/results",
        type=str,
        help="Path to the output directory, where the results will be saved",
    )
    parser.add_argument(
        "--model", default=MODEL, type=str, help="Specifies the model to use"
    )

    parser.add_argument(
        "--sequence_length",
        default=512,
        type=int,
        help="Max input sequence length for the model",
    )

    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting the benchmark...")
    logger.info(f"Arguments: {args}")

    preprocessor = Preprocessor(sequence_length=int(args.sequence_length))
    sentiment_analyser = SentimentAnalyser(model=args.model, preprocessor=preprocessor)

    data = preprocessor.read_in_data(data_path=args.data)

    sentiment_analyser.benchmark(data, output_dir=args.output_dir)

    logger.info("Benchmark completed successfully!")
