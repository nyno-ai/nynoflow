from abc import ABC, abstractmethod

from attrs import define, field

from nynoflow.chats import ChatMessage
from nynoflow.tokenizers import Tokernizers


@define(kw_only=True)
class BaseMemory(ABC):
    """Base class for memory backends.

    Attributes:
        chat_id (str): The chat id.
        persist (bool): Whether to persist the memory or not after each run. If False, the memory will be
                        deleted when the flow is finished.
        message_history (list[ChatMessage]): The message history. Stored in memory and backend.
        temporary_message_history (list[ChatMessage]): The temporary message history. Stored in memory only and used
                                                   for temporary messages like fixing messages.
    """

    chat_id: str = field()
    persist: bool = field(default=True)

    message_history: list[ChatMessage] = field(factory=list[ChatMessage])

    def __attrs_post_init__(self) -> None:
        """Load the message history."""
        self.load_message_history()

    def get_message_history_upto_token_limit(
        self, token_limit: int, tokenizer: Tokernizers
    ) -> list[ChatMessage]:
        """Get as much messages from the message history starting from most recent message that fit the token limit.

        Args:
            token_limit (int): The token limit.
            tokenizer (Tokernizers): The tokenizer to use.

        Returns:
            list[ChatMessage]: The message history by tokens.
        """
        message_history = list[ChatMessage]()
        token_count = 0
        for msg in reversed(self.message_history):
            token_count += tokenizer.token_count(list[ChatMessage]([msg]))
            if token_count > token_limit:
                break
            message_history.append(msg)
        return message_history

    @abstractmethod
    def load_message_history(self) -> None:
        """Load a chat from backend to memory."""

    @abstractmethod
    def _insert_message_backend(self, msg: ChatMessage) -> None:
        """Insert a message into the backend.

        Args:
            msg (ChatMessage): The message to insert.
        """

    def _insert_message_batch_backend(self, msgs: list[ChatMessage]) -> None:
        """Insert a batch of messages into the backend. Override if backend supports batch insertion.

        Args:
            msgs (list[ChatMessage]): The messages to insert.
        """
        for msg in msgs:
            self._insert_message_backend(msg)

    @abstractmethod
    def _remove_message_backend(self, msg: ChatMessage) -> None:
        """Remove a message from the backend.

        Args:
            msg (ChatMessage): The message to remove.
        """

    def insert_message(self, msg: ChatMessage) -> None:
        """Insert a message into the message history in both the message_history attribute and the backend.

        Args:
            msg (ChatMessage): The message to insert.
        """
        self.message_history.append(msg)
        self._insert_message_backend(msg)

    def insert_message_batch(self, msgs: list[ChatMessage]) -> None:
        """Insert a batch of messages into the message history in both the message_history attribute and the backend.

        Args:
            msgs (list[ChatMessage]): The messages to insert.
        """
        self.message_history.extend(msgs)
        self._insert_message_batch_backend(msgs)

    def remove_message(self, msg: ChatMessage) -> None:
        """Remove a message from the message history in both the message_history attribute and the backend.

        Args:
            msg (ChatMessage): The message to remove.
        """
        self.message_history.remove(msg)
        self._remove_message_backend(msg)

    def clean_temporary_message_history(self) -> None:
        """Clean the temporary message history."""
        for msg in self.message_history[:]:
            if msg.temporary:
                self.remove_message(msg)

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the memory. Called when the memory instance is deleted."""

    def __del__(self) -> None:
        """Delete the memory if persist is False."""
        if not self.persist:
            self.cleanup()
