import os
import tempfile
from typing import Generator
from uuid import uuid4

import pytest

from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.memory import LocalFileMemory, MemoryProviders
from tests.conftest import ConfigTests
from tests.memory.base_file_memory_tests import BaseFileMemoryTest


@pytest.fixture(scope="function")
def memory(config: ConfigTests) -> Generator[LocalFileMemory, None, None]:
    """Return the memory provider with unique configuration each call.

    This is used to test the memory provider with different configurations.
    """
    yield LocalFileMemory(chat_id=str(uuid4()), persist=False)


class TestLocalFileMemory(BaseFileMemoryTest):
    """Test memory implementations."""

    def is_memory_file_exists(self, memory: MemoryProviders) -> bool:
        """Check if the memory file exists."""
        # Needed for type checking.
        assert isinstance(memory, LocalFileMemory)
        return os.path.exists(memory.file_path)

    def test_custom_filepath(self) -> None:
        """Test using the memory with a custom path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "memory.json")
            memory = LocalFileMemory(
                chat_id=str(uuid4()), file_path=filepath, persist=False
            )
            msg0 = ChatMessage(
                provider_id="chatgpt",
                role="user",
                content="What is the captial of italy?",
            )
            memory.insert_message(msg0)
            assert os.path.exists(filepath)
            assert len(memory.message_history) == 1
