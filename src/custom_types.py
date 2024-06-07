from enum import Enum


class SentimentType(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0
    NOT_UNDERSTOOD = 2
