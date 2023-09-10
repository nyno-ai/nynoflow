import json
import os
import time

import cattrs
from attrs import define, field

from nynoflow.chats import ChatMessage
from nynoflow.memory.base_memory import BaseMemory


@define
class LocalFileMemoryStructure:
    """Structure of the local file memory."""

    chat_id: str
    messages: list[ChatMessage] = field(factory=list[ChatMessage])
    created_at: float = field(factory=time.time)
    updated_at: float = field(factory=time.time)


@define(kw_only=True)
class LocalFileMemory(BaseMemory):
    """Store message history in a local file."""

    file_path: str = field()

    @file_path.default
    def _default_file_path(self) -> str:
        return os.path.join(".", ".nynoflow", str(self.chat_id), "memory.json")

    def __attrs_post_init__(self) -> None:
        """Set the file path."""
        self.load_message_history()

    def _insert_message_backend(self, msg: ChatMessage) -> None:
        """Insert a new chat message into the json file.

        Args:
            msg (ChatMessage): The message to insert.
        """
        with open(self.file_path) as f:
            data_json = json.load(f)
            data = cattrs.structure(data_json, LocalFileMemoryStructure)

        data.messages.append(msg)

        with open(self.file_path, "w") as f:
            json.dump(cattrs.unstructure(data), f)

    def _insert_message_batch_backend(self, msgs: list[ChatMessage]) -> None:
        """Insert a batch of messages into the json file.

        Args:
            msgs (list[ChatMessage]): The messages to insert.
        """
        with open(self.file_path) as f:
            data = cattrs.structure(json.load(f), LocalFileMemoryStructure)

        data.messages.extend(msgs)

        with open(self.file_path, "w") as f:
            json.dump(cattrs.unstructure(data), f)

    def load_message_history(self) -> None:
        """Load a chat from backend to memory."""
        # Memory file exists
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                messages = json.load(f)["messages"]
            self.message_history = list[ChatMessage](
                cattrs.structure(msg, ChatMessage) for msg in messages
            )
        # Memory file does not exist, initalize memory file and directory
        else:
            data = LocalFileMemoryStructure(chat_id=self.chat_id)
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(cattrs.unstructure(data), f)

    def _remove_message_backend(self, msg: ChatMessage) -> None:
        """Remove a message from the backend.

        Args:
            msg (ChatMessage): The message to remove.
        """
        with open(self.file_path) as f:
            data = cattrs.structure(json.load(f), LocalFileMemoryStructure)

        data.messages.remove(msg)

        with open(self.file_path, "w") as f:
            json.dump(cattrs.unstructure(data), f)
