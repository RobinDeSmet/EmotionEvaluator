"""Evaulator API"""

import logging
import os
from http import HTTPStatus

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.preprocessor import Preprocessor
from src.sentiment_analyser import SentimentAnalyser
from src.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

load_dotenv()
MODEL = os.getenv("MODEL")
INPUT_TEXT_MAX_LENGTH = int(os.getenv("INPUT_TEXT_MAX_LENGTH"))

app = FastAPI()
preprocessor = Preprocessor()
sentiment_analyser = SentimentAnalyser(model=MODEL)


class Review(BaseModel):
    """Review object to be evaluated"""

    content: str = Field(..., max_length=INPUT_TEXT_MAX_LENGTH)


@app.get("/")
def read_root():
    """Generic root endpoint"""
    return {"Hello": "World"}


@app.post("/evaluate/")
async def evaluate_text(review: Review):
    """Evaluate the sentiment of a given review.

    Args:
        review (Review): The review to be evaluated.
    """
    logger.info("Evaulating the sentiment of the review...")

    try:
        # Preprocess the text
        text = preprocessor.preprocess_text(review.content, use_autocorreect=True)

        # Analyse the sentiment
        sentiment = sentiment_analyser.get_sentiment(text)

        # Return the sentiment
        logger.info("Sentiment evaluated successfully")
        return {"sentiment": sentiment}
    except Exception as exc:
        # Process the error
        logger.error(f"Error: {exc}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error": "An error occurred while evaluating the sentiment"},
        )
