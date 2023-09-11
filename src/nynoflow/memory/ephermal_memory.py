from attrs import define

from nynoflow.chats import ChatMessage
from nynoflow.memory.base_memory import BaseMemory


@define(kw_only=True)
class EphermalMemory(BaseMemory):
    """Store messages in memory."""

    def load_message_history(self) -> None:
        """Load a chat from backend to memory. Not relevant because this memory is ephermal."""

    def _insert_message_backend(self, msg: ChatMessage) -> None:
        """Not relevant because the ephermal backend is the memory itself.

        Args:
            msg (ChatMessage): The message to insert.
        """

    def _remove_message_backend(self, msg: ChatMessage) -> None:
        """Not relevant because the ephermal backend is the memory itself.

        Args:
            msg (ChatMessage): The message to remove.
        """

    def cleanup(self) -> None:
        """Cleanup the memory. Not relevant because this memory is ephermal."""
