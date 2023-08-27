from typing import Union

import openai
from attrs import asdict, define, field, frozen

from nynoflow.util import logger


@frozen
class ChatgptResponseUsage:
    """This is the usage object for the ChatGPT API."""

    prompt_tokens: int = field()
    completion_tokens: int = field()
    total_tokens: int = field()


@frozen
class ChatgptResponseFunctionCall:
    """This is the function call object for the ChatGPT API."""

    name: str = field()
    arguments: str = field()


@frozen
class ChatgptResponseMessage:
    """This is the message object for the ChatGPT API."""

    role: str = field()
    content: Union[str, None] = field(default=None)
    function_call: Union[ChatgptResponseFunctionCall, None] = field(default=None)


@frozen
class ChatgptResponseChoice:
    """This is the choice object for the ChatGPT API."""

    index: int = field()
    message: ChatgptResponseMessage = field()
    finish_reason: str = field()


@frozen
class ChatgptResponse:
    """This is the object returned by the OpenAI API for a chat completion.

    Parameters are taken from here: https://platform.openai.com/docs/api-reference/chat/object
    """

    id: str = field()
    object: str = field()
    created: int = field()
    model: str = field()
    choices: list[ChatgptResponseChoice] = field()
    usage: ChatgptResponseUsage = field()


@frozen
class ChatgptRequestFunction:
    """This is the function object for the ChatGPT API."""

    name: str = field()
    description: str = field()
    parameters: object = field()


@frozen
class ChatgptRequestFunctionCall:
    """This is the function call object for the ChatGPT API."""

    name: str = field()
    arguments: str = field()


@frozen
class ChatgptRequestMessage:
    """This is the message object for the ChatGPT API."""

    role: str = field()
    content: str = field()
    name: Union[str, None] = field(default=None)
    function_call: Union[ChatgptRequestFunctionCall, None] = field(default=None)


@frozen
class ChatgptRequest:
    """This is the request object for chatgpt completions.

    Parameters are taken from herer: https://platform.openai.com/docs/api-reference/chat/create?lang=python
    """

    messages: list[ChatgptRequestMessage] = field()


@define
class ChatgptProvider:
    """LLM Engine for OpenAI ChatGPT."""

    organization: str = field()
    api_key: str = field()
    model: str = field()

    functions: Union[list[ChatgptRequestFunction], None] = field(default=None)
    function_call: Union[str, None] = field(default=None)
    temperature: Union[float, None] = field(default=None)
    top_p: Union[float, None] = field(default=None)
    n: Union[int, None] = field(default=None)
    stream = False  # Streams not supported in this library yet
    stop: Union[str, list[str], None] = field(default=None)
    max_tokens: Union[int, None] = field(default=None)
    presence_penalty: Union[float, None] = field(default=None)
    frequency_penalty: Union[float, None] = field(default=None)
    logit_bias: Union[dict[str, float], None] = field(default=None)
    user: Union[str, None] = field(default=None)

    provider_id: str = field(default="chatgpt")

    def __attrs_post_init__(self) -> None:
        """Configure the openai package with auth and configurations."""
        logger.debug("Configuring ChatGPT Provider")
        openai.api_key = self.api_key
        openai.organization = self.organization

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
        res: ChatgptResponse = openai.ChatCompletion.create(**asdict(req))
        content: str = res.choices[0].message.content or ""  # TODO fix this hack
        return content
