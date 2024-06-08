from enum import Enum


class SentimentType(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0
    NOT_UNDERSTOOD = 2

    def __str__(self):
        if self == SentimentType.POSITIVE:
            return "Positive"
        elif self == SentimentType.NEGATIVE:
            return "Negative"
        elif self == SentimentType.NEUTRAL:
            return "Neutral"
        elif self == SentimentType.NOT_UNDERSTOOD:
            return "Not Understood"
        else:
            return super().__str__()
