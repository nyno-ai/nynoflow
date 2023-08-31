from abc import ABC, abstractmethod


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

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens.

        Args:
            tokens (list[int]): The tokens to decode.

        Returns:
            str: The decoded tokens.
        """

    def token_count(self, text: str) -> int:
        """Get the number of tokens in a string.

        Args:
            text (str): The text to get the number of tokens of.

        Returns:
            int: The number of tokens in the string.
        """
        return len(self.encode(text))
