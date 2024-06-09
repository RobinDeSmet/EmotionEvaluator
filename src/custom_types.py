"""Custom types"""

from enum import Enum


class SentimentType(Enum):
    """Sentiment types"""

    POSITIVE = 1
    NEGATIVE = 0
    NEUTRAL = 2
    NOT_UNDERSTOOD = 3

    def __str__(self):
        if self == SentimentType.POSITIVE:
            return "Positive"
        if self == SentimentType.NEGATIVE:
            return "Negative"
        if self == SentimentType.NEUTRAL:
            return "Neutral"
        if self == SentimentType.NOT_UNDERSTOOD:
            return "Not Understood"
        return super().__str__()
