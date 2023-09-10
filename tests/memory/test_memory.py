import os
import tempfile
from uuid import uuid4

from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.memory import LocalFileMemory


class TestMemory:
    """Test memory implementations."""

    def test_local_file_memory_basic_operations(self) -> None:
        """Test the local file memory."""
        memory = LocalFileMemory(chat_id=str(uuid4()), persist=False)
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

    def test_local_file_memory_load(self) -> None:
        """Test the local file memory."""
        chat_id = str(uuid4())
        memory = LocalFileMemory(chat_id=chat_id)
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
        del memory

        memory = LocalFileMemory(chat_id=chat_id)
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

        del memory
        memory = LocalFileMemory(chat_id=chat_id, persist=False)
        assert len(memory.message_history) == 3

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

    def test_persistence_and_cleanup(self) -> None:
        """Test persistence and cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "memory.json")
            memory = LocalFileMemory(chat_id=str(uuid4()), file_path=filepath)
            msg0 = ChatMessage(
                provider_id="chatgpt",
                role="user",
                content="What is the captial of italy?",
            )
            memory.insert_message(msg0)
            assert os.path.exists(filepath)
            assert len(memory.message_history) == 1
            del memory
            assert os.path.exists(filepath)

            memory = LocalFileMemory(chat_id=str(uuid4()), file_path=filepath)
            assert len(memory.message_history) == 1
            memory.cleanup()
            assert not os.path.exists(filepath)
