import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")


def url(path: str) -> str:
    """Generates a URL from a path.

    Args:
        path (str): The path to the resource.

    Returns:
        str: The URL to the resource.
    """
    return f"{BASE_URL}/{path}"
