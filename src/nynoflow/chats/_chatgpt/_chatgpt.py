from typing import Union
from warnings import warn

import openai
from attrs import define, field

from nynoflow.chats._chatgpt._chatgpt_objects import (
    ChatgptMessageHistory,
    ChatgptRequest,
    ChatgptRequestMessage,
    ChatgptResponse,
)
from nynoflow.chats.chat_objects import ChatMessageHistory
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
    # n: Union[int, None] = field(default=None) # Make sure it isn't more then 1
    # stream = False  # Streams not supported in this library yet
    # stop: Union[str, list[str], None] = field(default=None)
    # max_tokens: Union[int, None] = field(default=None)
    # presence_penalty: Union[float, None] = field(default=None)
    # frequency_penalty: Union[float, None] = field(default=None)
    # logit_bias: Union[dict[str, float], None] = field(default=None)
    # user: Union[str, None] = field(default=None)

    provider_id: str = field(default="chatgpt")
    openai_chat_completion_client: openai.ChatCompletion = field(init=False)
    token_limit: Union[int, None] = field(default=None)
    tokenizer: ChatgptTiktokenTokenizer = field(
        init=False
    )  # Used to override the default tokenizer

    def __attrs_post_init__(self) -> None:
        """Configure the openai package with auth and configurations."""
        self.openai_chat_completion_client = openai.ChatCompletion(
            api_key=self.api_key,
            organization=self.organization,
        )

        self.tokenizer = ChatgptTiktokenTokenizer(self.model)

        if self.token_limit is None:
            self.token_limit = self._model_token_limit()

    def _model_token_limit(
        self,
    ) -> int:  # pragma: no cover # Ignored because no use testing each model separately
        """Get the token limit for the model.

        Raises:
            NotImplementedError: If the model is not supported.

        Returns:
            int: The token limit.
        """
        # Warn the user if he uses a model without a specific version, and alert him to the model being assumed.
        model_for_token_limit = self.model
        if self.model == "gpt-3.5-turbo":
            warn(
                "Model gpt-3.5-turbo may change in the future, and it is recommended to"
                "use a specific version of the model instead. Using token limit for model gpt-3.5-turbo-0613 instead."
            )
            model_for_token_limit = "gpt-3.5-turbo-0613"  # noqa: S105
        elif self.model == "gpt-4":
            warn(
                "Model gpt-4 may change in the future, and it is recommended to"
                "use a specific version of the model instead. Using token limit for model gpt-4-0613 instead."
            )
            model_for_token_limit = "gpt-4-0613"  # noqa: S105

        # Get the token limit for the model
        if model_for_token_limit in {"gpt-4-32k-0613", "gpt-4-32k-0314"}:
            return 32768

        if model_for_token_limit in {"gpt-4-0314", "gpt-4-0613"}:
            return 8192

        if model_for_token_limit in {"gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301"}:
            return 4096

        if model_for_token_limit in {"gpt-3.5-turbo-16k-0613"}:
            return 16384

        raise NotImplementedError(
            f"Token limit not implemented for model {model_for_token_limit}. Please open an issue on GitHub."
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
        return self.tokenizer.token_count(converted_message_history)

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
            ChatgptRequestMessage(role=msg["role"], content=msg["content"])
            for msg in message_history
        ]

    def completion(self, messages: ChatMessageHistory) -> str:
        """Get a completion from the OpenAI API.

        Args:
            messages (ChatMessageHistory): The message history to prompt with.

        Returns:
            The completion.
        """
        chatgpt_messages: ChatgptMessageHistory = self._convert_message_history(
            messages
        )
        req: ChatgptRequest = ChatgptRequest(messages=chatgpt_messages)
        res: ChatgptResponse = self.openai_chat_completion_client.create(**req)

        content: str = res["choices"][0]["message"]["content"]
        return content
