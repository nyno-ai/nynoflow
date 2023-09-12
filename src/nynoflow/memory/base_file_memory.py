import json
import time
from abc import abstractmethod

import cattrs
from attrs import define, field

from nynoflow.chats import ChatMessage
from nynoflow.memory.base_memory import BaseMemory


@define
class FileMemoryStructure:
    """Structure of the local file memory."""

    chat_id: str
    messages: list[ChatMessage] = field(factory=list[ChatMessage])
    created_at: float = field(factory=time.time)
    updated_at: float = field(factory=time.time)


@define(kw_only=True)
class BaseFileMemory(BaseMemory):
    """Store message history in a local file."""

    @abstractmethod
    def _read_memory_file(self) -> str:
        """Read the memory file. Raise a FileNotFoundError if the file does not exist."""

    @abstractmethod
    def _write_memory_file(self, content: str) -> None:
        """Write to the memory file. Create the file if it does not exist."""

    @abstractmethod
    def _remove_memory_file(self) -> None:
        """Remove the memory file."""

    def cleanup(self) -> None:
        """Cleanup the memory."""
        self._remove_memory_file()
        self.message_history = list[ChatMessage]()

    def load_message_history(self) -> None:
        """Load a chat from backend to memory. Initialize the memory file if it does not exist."""
        # Memory file exists
        try:
            data_json = json.loads(self._read_memory_file())
            data = cattrs.structure(data_json, FileMemoryStructure)
            self.message_history = data.messages

        except FileNotFoundError:
            data = FileMemoryStructure(chat_id=self.chat_id)
            self._write_memory_file(json.dumps(cattrs.unstructure(data)))

    def _insert_message_backend(self, msg: ChatMessage) -> None:
        """Insert a new chat message into the json file.

        Args:
            msg (ChatMessage): The message to insert.
        """
        data_json = json.loads(self._read_memory_file())
        data = cattrs.structure(data_json, FileMemoryStructure)
        data.messages.append(msg)
        self._write_memory_file(json.dumps(cattrs.unstructure(data)))

    def _insert_message_batch_backend(self, msgs: list[ChatMessage]) -> None:
        """Insert a batch of messages into the json file.

        Args:
            msgs (list[ChatMessage]): The messages to insert.
        """
        data_json = json.loads(self._read_memory_file())
        data = cattrs.structure(data_json, FileMemoryStructure)
        data.messages.extend(msgs)
        self._write_memory_file(json.dumps(cattrs.unstructure(data)))

    def _remove_message_backend(self, msg: ChatMessage) -> None:
        """Remove a message from the backend.

        Args:
            msg (ChatMessage): The message to remove.
        """
        data_json = json.loads(self._read_memory_file())
        data = cattrs.structure(data_json, FileMemoryStructure)
        data.messages.remove(msg)
        self._write_memory_file(json.dumps(cattrs.unstructure(data)))
