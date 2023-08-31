from typing import TypedDict, Union


class ChatgptResponseUsage(TypedDict):
    """This is the usage object for the ChatGPT API."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatgptResponseMessage(TypedDict, total=False):
    """This is the message object for the ChatGPT API."""

    role: str
    content: Union[str, None]


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
    choices: list[ChatgptResponseChoice]
    usage: ChatgptResponseUsage


class ChatgptRequestMessage(TypedDict, total=False):
    """This is the message object for the ChatGPT API."""

    role: str
    content: Union[str, None]
    name: Union[str, None]


class ChatgptRequest(TypedDict, total=False):
    """This is the request object for chatgpt completions."""

    messages: list[ChatgptRequestMessage]


ChatgptMessageHistory = list[ChatgptRequestMessage]
