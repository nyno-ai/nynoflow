from abc import ABC, abstractmethod

from nynoflow.chats.chat_objects import ChatMessage


class BaseTokenizer(ABC):
    """Base tokenizer class with abstract methods for tokenization and detokenization."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode a string.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: The coded text.
        """

    def token_count(self, messages: list[ChatMessage]) -> int:
        """Get the number of tokens in a string.

        Args:
            messages (list[ChatMessage]): The text to get the number of tokens of.

        Returns:
            int: The number of tokens in the string.
        """
        return len(self.encode(str(messages)))
