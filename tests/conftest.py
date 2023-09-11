import os
from typing import TypedDict

import pytest
from dotenv import load_dotenv


class ConfigTests(TypedDict):
    """Config for testing."""

    OPENAI_API_KEY: str
    AWS_S3_BUCKET_NAME: str


@pytest.fixture(autouse=True, scope="session")
def config() -> ConfigTests:
    """Load the test environment variables.

    Returns:
        ConfigTests: The test config.
    """
    load_dotenv(".env.test")

    config = ConfigTests(
        {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "AWS_S3_BUCKET_NAME": os.environ["AWS_S3_BUCKET_NAME"],
        }
    )
    return config
