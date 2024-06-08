import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from http import HTTPStatus
from pydantic import BaseModel, Field

from src.preprocessor import Preprocessor
from src.sentiment_analyser import SentimentAnalyser
from src.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

load_dotenv()
MODEL = os.getenv("MODEL")

app = FastAPI()
preprocessor = Preprocessor()
sentiment_analyser = SentimentAnalyser(model=MODEL)


class Review(BaseModel):
    content: str = Field(..., max_length=512)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/evaluate/")
async def evaluate_text(review: Review):
    logger.info(f"Evaulating the sentiment of the review...")

    try:
        # Preprocess the text
        text = preprocessor.preprocess_text(review.content)

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
