from typing import List, TypedDict, Union

import openai
from attrs import define, field

from nynoflow.util import logger


class ChatgptResponseUsage(TypedDict):
    """This is the usage object for the ChatGPT API."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatgptResponseFunctionCall(TypedDict):
    """This is the function call object for the ChatGPT API."""

    name: str
    arguments: str


class ChatgptResponseMessage(TypedDict, total=False):
    """This is the message object for the ChatGPT API."""

    role: str
    content: Union[str, None]
    function_call: Union[ChatgptResponseFunctionCall, None]


class ChatgptResponseChoice(TypedDict):
    """This is the choice object for the ChatGPT API."""

    index: int
    message: ChatgptResponseMessage
    finish_reason: str


class ChatgptResponse(TypedDict):
    """This is the object returned by the OpenAI API for a chat completion."""

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatgptResponseChoice]
    usage: ChatgptResponseUsage


class ChatgptRequestFunction(TypedDict):
    """This is the function object for the ChatGPT API."""

    name: str
    description: str
    parameters: object


class ChatgptRequestFunctionCall(TypedDict):
    """This is the function call object for the ChatGPT API."""

    name: str
    arguments: str


class ChatgptRequestMessage(TypedDict, total=False):
    """This is the message object for the ChatGPT API."""

    role: str
    content: str
    name: Union[str, None]
    function_call: Union[ChatgptRequestFunctionCall, None]


class ChatgptRequest(TypedDict):
    """This is the request object for chatgpt completions."""

    messages: List[ChatgptRequestMessage]


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
    openai_chat_completion_client: openai.ChatCompletion = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the openai package with auth and configurations."""
        logger.debug("Configuring ChatGPT Provider")
        self.openai_chat_completion_client = openai.ChatCompletion(
            api_key=self.api_key,
            organization=self.organization,
        )

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
