from typing import Union

import openai
from attrs import define, field

from nynoflow.chats._chatgpt._chatgpt_objects import (
    ChatgptMessageHistory,
    ChatgptRequest,
    ChatgptRequestMessage,
    ChatgptResponse,
)
from nynoflow.chats.chat_objects import ChatMessageHistory
from nynoflow.util import logger
from nynoflow.utils.tokenizers.openai_tokenizer import ChatgptTiktokenTokenizer


@define
class ChatgptProvider:
    """LLM Engine for OpenAI ChatGPT."""

    organization: str = field()
    api_key: str = field()
    model: str = field()

    # functions: Union[list[ChatgptRequestFunction], None] = field(default=None)
    # function_call: Union[str, None] = field(default=None)
    # temperature: Union[float, None] = field(default=None)
    # top_p: Union[float, None] = field(default=None)
    # n: Union[int, None] = field(default=None)
    # stream = False  # Streams not supported in this library yet
    # stop: Union[str, list[str], None] = field(default=None)
    # max_tokens: Union[int, None] = field(default=None)
    # presence_penalty: Union[float, None] = field(default=None)
    # frequency_penalty: Union[float, None] = field(default=None)
    # logit_bias: Union[dict[str, float], None] = field(default=None)
    # user: Union[str, None] = field(default=None)

    provider_id: str = field(default="chatgpt")
    openai_chat_completion_client: openai.ChatCompletion = field(init=False)
    tokenizer: Union[ChatgptTiktokenTokenizer, None] = field(
        default=None
    )  # Used to override the default tokenizer

    def __attrs_post_init__(self) -> None:
        """Configure the openai package with auth and configurations."""
        logger.debug("Configuring ChatGPT Provider")
        self.openai_chat_completion_client = openai.ChatCompletion(
            api_key=self.api_key,
            organization=self.organization,
        )

        if self.tokenizer is None:
            self.tokenizer = ChatgptTiktokenTokenizer(self.model)

    @property
    def _token_limit(self) -> int:
        """Get the token limit for the model.

        Raises:
            NotImplementedError: If the model is not supported.

        Returns:
            int: The token limit.
        """
        if self.model in {"gpt-4-32k-0613", "gpt-4-32k-0314"}:
            return 32768

        if self.model in {"gpt-4-0314", "gpt-4-0613"}:
            return 8192

        if self.model in {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301"}:
            return 4096

        if self.model in {"gpt-3.5-turbo-16k-0613"}:
            return 16384

        raise NotImplementedError(
            f"Token limit not implemented for model {self.model}. Please open an issue on GitHub."
        )

    def _num_tokens(self, messages: ChatMessageHistory) -> int:
        """Get the number of tokens in a message list for this provider and tokenizer.

        Args:
            messages: List of messages

        Returns:
            The number of tokens in the message list.
        """
        converted_message_history: ChatgptMessageHistory = (
            self._convert_message_history(messages)
        )
        return self.tokenizer.token_count(converted_message_history, self.model)

    def _convert_message_history(
        self, message_history: ChatMessageHistory
    ) -> ChatgptMessageHistory:
        """Convert the message history to a ChatGPT format message history.

        Args:
            message_history (ChatMessageHistory): The message history to convert.

        Returns:
            ChatgptMessageHistory: The converted message history.
        """
        return [
            ChatgptRequestMessage(role=m.role, content=m.content)
            for m in message_history
        ]

    def completion(self, prompt: str) -> str:
        """Get a completion from the OpenAI API.

        Args:
            prompt (str): The prompt to use for the completion.

        Returns:
            The completion.
        """
        req: ChatgptRequest = ChatgptRequest(
            messages=[ChatgptRequestMessage(role="user", content=prompt)]
        )
        res: ChatgptResponse = self.openai_chat_completion_client.create(**req)
        content: str = (
            res["choices"][0]["message"]["content"] or ""
        )  # TODO fix this hack
        return content
