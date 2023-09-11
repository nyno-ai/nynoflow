"""This module is named without "test" prefix so pytest does not try to run it as a test.

This module contains a base class for testing file based memory providers.
It does not contain any tests itself that should run on its own.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.memory import MemoryProviders


MemoryProviderType = TypeVar("MemoryProviderType", bound=MemoryProviders)


class BaseMemoryTest(ABC):
    """Generic test class for all file based memory providers."""

    @abstractmethod
    def is_backend_memory_exists(self, memory: MemoryProviderType) -> bool:
        """Check if the memory file exists."""

    def test_local_file_memory_basic_operations(
        self, memory: MemoryProviderType
    ) -> None:
        """Test the local file memory."""
        print(memory.chat_id)
        msg0 = ChatMessage(
            provider_id="chatgpt",
            role="user",
            content="What is the captial of italy?",
        )
        msg1 = ChatMessage(
            provider_id="chatgpt",
            role="assistant",
            content="Rome.",
        )
        msg2 = ChatMessage(
            provider_id="chatgpt",
            role="user",
            content="What is the captial of france?",
        )
        memory.insert_message_batch([msg0, msg1])

        assert len(memory.message_history) == 2
        assert memory.message_history[0] == msg0
        assert memory.message_history[1] == msg1

        memory.insert_message(msg2)
        assert len(memory.message_history) == 3
        assert memory.message_history[2] == msg2

        memory.remove_message(msg1)
        assert len(memory.message_history) == 2
        assert memory.message_history[0] == msg0
        assert memory.message_history[1] == msg2

    def test_local_file_memory_load(self, memory: MemoryProviderType) -> None:
        """Test the local file memory."""
        print(memory.chat_id)
        memory.insert_message_batch(
            [
                ChatMessage(
                    provider_id="chatgpt",
                    role="user",
                    content="What is the captial of italy?",
                ),
                ChatMessage(
                    provider_id="chatgpt",
                    role="assistant",
                    content="Rome.",
                ),
            ]
        )
        assert len(memory.message_history) == 2

        # Reset the memory and load it again
        memory.message_history = list[ChatMessage]()
        memory.load_message_history()

        memory.insert_message_batch(
            [
                ChatMessage(
                    provider_id="chatgpt",
                    role="user",
                    content="What is the captial of france?",
                ),
                ChatMessage(
                    provider_id="chatgpt",
                    role="assistant",
                    content="Paris.",
                ),
            ]
        )

        assert len(memory.message_history) == 4
        assert memory.message_history[0].content == "What is the captial of italy?"
        assert memory.message_history[1].content == "Rome."
        assert memory.message_history[2].content == "What is the captial of france?"
        assert memory.message_history[3].content == "Paris."

        memory.remove_message(memory.message_history[0])

        # Reset the memory and load it again
        memory.message_history = list[ChatMessage]()
        memory.load_message_history()

        assert len(memory.message_history) == 3
        assert memory.message_history[0].content == "Rome."
        assert memory.message_history[1].content == "What is the captial of france?"
        assert memory.message_history[2].content == "Paris."

    def test_cleanup(self, memory: MemoryProviderType) -> None:
        """Test persistence and cleanup."""
        msg0 = ChatMessage(
            provider_id="chatgpt",
            role="user",
            content="What is the captial of italy?",
        )
        memory.insert_message(msg0)
        assert self.is_backend_memory_exists(memory)
        assert len(memory.message_history) == 1

        # Save configuration to check if the file exists after cleanup
        memory.cleanup()
        assert not self.is_backend_memory_exists(memory)
