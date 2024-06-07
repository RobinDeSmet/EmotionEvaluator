import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.preprocessor import Preprocessor
from src.sentiment_analyser import SentimentAnalyser
from src.utils import configure_logging

app = FastAPI()
preprocessor = Preprocessor()
sentiment_analyser = SentimentAnalyser()

configure_logging()
logger = logging.getLogger(__name__)


class Review(BaseModel):
    content: str = Field(..., max_length=512)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/evaulate/")
async def evaluate_text(review: Review):
    logger.info(f"Evaulating the sentiment of the review...")
    # Preprocess the text
    text = preprocessor.preprocess_text(review.content)

    # Analyse the sentiment
    sentiment = sentiment_analyser.get_sentiment(text)

    logger.info("Sentiment evaluated successfully")
    return {"sentiment": sentiment}
