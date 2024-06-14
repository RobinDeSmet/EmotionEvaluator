"""Test for the sentiment analyzer"""

from src.sentiment_analyser import SentimentAnalyser
from src.custom_types import SentimentType

preprocessor = SentimentAnalyser()


def test_evaluate_get_sentiment_works():
    """Test if the get_sentiment function works correctly."""
    test_data_pos = "Such a great movie. I loved the acting!"
    test_data_neg = "I hated this movie!"

    # Get the sentiment the text
    sentiment_pos = preprocessor.get_sentiment(test_data_pos)
    sentiment_neg = preprocessor.get_sentiment(test_data_neg)

    # Check the results
    assert sentiment_pos == SentimentType.POSITIVE.value
    assert sentiment_neg == SentimentType.NEGATIVE.value
