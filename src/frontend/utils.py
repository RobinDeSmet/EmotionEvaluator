"""This module contains utility functions for the frontend."""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BASE_URL")


def get_url(path: str) -> str:
    """Generates a URL from a path.

    Args:
        path (str): The path to the resource.

    Returns:
        str: The URL to the resource.
    """
    url = f"{BASE_URL}{path}"

    logger.info(f"Trying to connect to following resourcse: {url}...")

    return url
