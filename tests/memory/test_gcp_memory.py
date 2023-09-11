import json
from typing import Generator
from uuid import uuid4

import pytest
from google.cloud.storage import Client  # type: ignore
from google.oauth2.service_account import Credentials  # type: ignore

from nynoflow.memory import GcpBlobMemory, MemoryProviders
from tests.conftest import ConfigTests
from tests.memory.base_file_memory_tests import BaseFileMemoryTest


@pytest.fixture(scope="function")
def memory(config: ConfigTests) -> Generator[GcpBlobMemory, None, None]:
    """Return a unique memory client on each call."""
    client = Client(
        credentials=Credentials.from_service_account_info(
            json.loads(config["GCP_SERVICE_ACCOUNT"])
        )
    )
    yield GcpBlobMemory(
        chat_id=str(uuid4()),
        gcp_client=client,
        bucket_name=config["GCP_BUCKET_NAME"],
        persist=False,
    )


class TestGCPBlobMemory(BaseFileMemoryTest):
    """Test memory implementations."""

    def is_memory_file_exists(self, memory: MemoryProviders) -> bool:
        """Check if the memory file exists."""
        # Needed for type checking.
        assert isinstance(memory, GcpBlobMemory)
        is_exists: bool = memory.blob.exists()
        return is_exists

    def test_gcp_bucket_name(self, config: ConfigTests) -> None:
        """Test that the GCP bucket name is set correctly."""
        chat_id = str(uuid4())
        memory = GcpBlobMemory(
            chat_id=chat_id,
            bucket_name=config["GCP_BUCKET_NAME"],
            gcp_client=Client(
                credentials=Credentials.from_service_account_info(
                    json.loads(config["GCP_SERVICE_ACCOUNT"])
                )
            ),
            persist=False,
        )
        assert memory.bucket_name == config["GCP_BUCKET_NAME"]
