from typing import Generator
from uuid import uuid4

import pytest
from azure.core.exceptions import ResourceNotFoundError

from nynoflow.memory import AzureBlobMemory, MemoryProviders
from tests.conftest import ConfigTests
from tests.memory.base_file_memory_tests import BaseFileMemoryTest


@pytest.fixture(scope="function")
def memory(config: ConfigTests) -> Generator[AzureBlobMemory, None, None]:
    """Return a unique memory client on each call."""
    yield AzureBlobMemory(
        chat_id=str(uuid4()),
        container_name=config["AZURE_CONTAINER_NAME"],
        connection_string=config["AZURE_CONNECTION_STRING"],
        persist=False,
    )


class TestAzureBlobMemory(BaseFileMemoryTest):
    """Test memory implementations."""

    def is_memory_file_exists(self, memory: MemoryProviders) -> bool:
        """Check if the memory file exists."""
        # Needed for type checking.
        assert isinstance(memory, AzureBlobMemory)
        try:
            memory._blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False

    def test_azure_client_with_parameters(self, config: ConfigTests) -> None:
        """Test that the Azure client is created with the correct parameters."""
        chat_id = str(uuid4())
        memory = AzureBlobMemory(
            chat_id=chat_id,
            container_name=config["AZURE_CONTAINER_NAME"],
            connection_string=config["AZURE_CONNECTION_STRING"],
            persist=False,
        )
        assert chat_id == memory.chat_id
