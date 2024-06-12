"""This module contains the Sentiment Analysier for our Emotion Evaluator application."""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from src.custom_types import SentimentType
from src.preprocessor import Preprocessor

load_dotenv()

logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH")
PREDICTED_SENTIMENT_COLUMN_NAME = os.getenv("PREDICTED_SENTIMENT_COLUMN_NAME")
MODEL = os.getenv("MODEL")


class SentimentAnalyser:
    """This class analyses the sentiment of a given text"""

    def __init__(
        self,
        model: str = MODEL,
        preprocessor: Preprocessor = Preprocessor(),
        framework="tf",
    ):
        logger.info("Initializing the sentiment analyser...")

        self.model = model

        self.framework = framework

        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            framework=self.framework,
        )

        self.preprocessor = preprocessor

        logger.info("Sentiment analyser initialized successfully")

    def set_model(self, model: str):
        """Sets the model to be used for sentiment analysis.

        Args:
            model (str): The model to be used.
        """
        logger.info(f"Setting the model to: {model}...")

        self.model = model

        self.pipeline = pipeline("sentiment-analysis", model=self.model)

        logger.info("Model set successfully")

    def get_sentiment(self, text: str) -> SentimentType:
        """Analyses the sentiment of the given text.
        Args:
            text (str): The text to preprocess.

        Returns:
            str: The sentiment of the text (either positive or negative).
        """
        logger.info(f"Analysing sentiment of the text: {text}")

        # Calculate the sentiment of the text
        pos_score = 0
        neg_score = 0

        sentiment = self.pipeline(text)

        logger.info(f"Sentiment from model: {sentiment}")

        if "STAR" in sentiment[0]["label"].upper():
            # Models who use 5 star system as sentiment labels
            for prediction in sentiment:
                predicted_sentiment = int(prediction["label"].split(" ")[0])
                predicted_score = prediction["score"]

                if predicted_sentiment > 3:
                    pos_score += predicted_score
                elif predicted_sentiment == 3:
                    pos_score += predicted_score * 0.1
                elif predicted_sentiment == 2:
                    neg_score += predicted_score
                else:
                    neg_score += 1.5 * predicted_score
        else:
            # Models who use descriptive labels as sentiment labels
            for prediction in sentiment:
                predicted_sentiment = prediction["label"].upper()
                predicted_sentiment_score = prediction["score"]

                if "POS" in predicted_sentiment or predicted_sentiment == "LABEL_1":
                    pos_score += predicted_sentiment_score
                elif "NEU" in predicted_sentiment:
                    pos_score += predicted_sentiment_score * 0.1
                else:
                    neg_score += predicted_sentiment_score

        if pos_score >= neg_score:
            sentiment = SentimentType.POSITIVE.value
            sentiment_log = SentimentType.POSITIVE
        else:
            sentiment = SentimentType.NEGATIVE.value
            sentiment_log = SentimentType.NEGATIVE

        logger.info(f"Resulting sentiment: {sentiment_log}")

        return sentiment

    def analyse(
        self, data: pd.DataFrame, column_to_process: str = "review"
    ) -> pd.DataFrame:
        """Analyses the sentiment of the given dataframe. And adds a new column with the sentiment (predicted_sentiment).

        Args:
            data (pd.DataFrame): The input dataframe.
            column_to_process (str, optional): The column that must be processed. Defaults to "review".

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        logger.info("Analysing the sentiment of the data...")

        data[PREDICTED_SENTIMENT_COLUMN_NAME] = data[column_to_process].apply(
            self.get_sentiment
        )

        logger.info("Data analysed successfully")

        return data

    def benchmark(
        self,
        data: pd.DataFrame,
        output_dir: str = "src/results",
    ):
        """Benchmarks the sentiment analyser on the given dataset.

        Args:
            data (pd.DataFrame): The dataset to benchmark on.
        """
        logger.info("Benchmarking the sentiment analyser...")

        logger.info("Setting up the output directory...")

        output_dir = os.path.join(output_dir, self.model.replace("/", "_"))

        # Check if the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Preprocessing the data...")

        data = self.preprocessor.preprocess(data)

        logger.info("Analyzing the data...")

        start = datetime.now()
        data = self.analyse(data)
        end = datetime.now()

        logger.info("Writing the results to a csv file...")

        data["review"] = data["review"].apply(" ".join)
        data.to_csv(os.path.join(output_dir, "results.csv"), index=False)

        logger.info("Generating the classification report...")

        # Generate the classification report
        report = classification_report(
            data["sentiment"], data[PREDICTED_SENTIMENT_COLUMN_NAME]
        )

        # Save the classification report to a file
        with open(
            f"{output_dir}/classification_report.txt", "w", encoding="utf-8"
        ) as f:
            f.write(f"[Amount of samples: {len(data)}] Generated in {end - start}s\n\n")
            f.write(report)

        logger.info("Writing misclassified reviews to a file...")

        # Get the misclassified reviews
        misclassified = data[data["sentiment"] != data[PREDICTED_SENTIMENT_COLUMN_NAME]]

        # Save the misclassified reviews to a file
        with open(
            f"{output_dir}/classification_report.txt", "a", encoding="utf-8"
        ) as f:
            f.write("\n\nMisclassified reviews:\n\n")
            for _, row in misclassified.iterrows():
                f.write(
                    f"[Predicted: {row[PREDICTED_SENTIMENT_COLUMN_NAME]}, Truth: {row['sentiment']}] Review: {row['review']}\n\n"
                )

        logger.info("Generating the confusion matrix...")

        # Mapping numeric labels to string labels
        label_mapping = {
            1: f"{SentimentType.POSITIVE}",
            0: f"{SentimentType.NEGATIVE}",
        }
        data["sentiment"] = data["sentiment"].map(label_mapping)
        data["predicted_sentiment"] = data["predicted_sentiment"].map(label_mapping)

        # Generate the confusion matrix
        cf_matrix = confusion_matrix(
            data["sentiment"],
            data[PREDICTED_SENTIMENT_COLUMN_NAME],
        )

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.unique(data[PREDICTED_SENTIMENT_COLUMN_NAME]),
            yticklabels=np.unique(data["sentiment"]),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        # Save the confusion matrix to a file
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

        logger.info("Sentiment analyser benchmarked successfully")
