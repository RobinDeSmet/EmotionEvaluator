from fastapi.testclient import TestClient
from http import HTTPStatus
from src.evaluator_api.main import app
from src.custom_types import SentimentType
import pytest

client = TestClient(app)


def test_evaluate_text_works():
    # Define a test review
    review = {"content": "This is a positive review!"}

    # Send a POST request to the /evaluate/ endpoint
    response = client.post("/evaluate/", json=review)

    # Check that the response status code is 200
    assert response.status_code == HTTPStatus.OK

    # Check that the response body contains a 'sentiment' key
    assert "sentiment" in response.json()

    # Optionally, check the sentiment value
    assert response.json()["sentiment"] == SentimentType.POSITIVE.value


def test_evaluate_text_fails():
    # Define an invalid test review
    review = {"contentsss": ""}

    # Send a POST request to the /evaluate/ endpoint
    response = client.post("/evaluate/", json=review)

    # Check that the response status code is 422
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
