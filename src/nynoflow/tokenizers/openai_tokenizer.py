from warnings import warn

import tiktoken
from attrs import define, field

from nynoflow.chats.chat_objects import ChatMessage


@define
class OpenAITokenizer:
    """Tokenizer for ChatGPT using TikToken.

    Args:
        model (str): The model to use.
    """

    model: str = field()

    _encoding: tiktoken.Encoding = field(init=False)
    _tokens_per_message: int = field(init=False)
    _tokens_per_name: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the TikToken tokenizer.

        Taken from here: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            warn(f"Model {self.model} not found. Using cl100k_base encoding.")
            self._encoding = tiktoken.get_encoding("cl100k_base")

        if self.model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            self._tokens_per_message = 4
            self._tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            self._tokens_per_message = 3
            self._tokens_per_name = 1

    def _calculate_messages_tokens(self, messages: list[ChatMessage]) -> int:
        """Return the number of tokens used by a list of messages.

        Args:
            messages (list[ChatMessage]): The messages to count the tokens of.

        Returns:
            int: The number of tokens used by the messages.
        """
        token_count = 0
        for message in messages:
            message_tokens = (
                self._tokens_per_message
                + len(self._encoding.encode(str(message.content)))
                + len(self._encoding.encode(str(message.role)))
            )
            token_count += message_tokens
        token_count += 3  # every reply is primed with <|start|>assistant<|message|>
        return token_count

    def token_count(
        self,
        messages: list[ChatMessage],
    ) -> int:
        """Return the number of tokens used by a list of messages and functions.

        Args:
            messages (list[ChatMessage]): The messages to count the tokens of.

        Returns:
            int: The number of tokens used by the messages.
        """
        # chatgpt_message_history = ChatgptMessageHistory(
        #     [
        #         ChatgptRequestMessage(
        #             role=msg.role,
        #             name=None,
        #             content=msg.content,
        #         )
        #         for msg in messages
        #     ]
        # )

        token_count = self._calculate_messages_tokens(messages)

        return token_count
