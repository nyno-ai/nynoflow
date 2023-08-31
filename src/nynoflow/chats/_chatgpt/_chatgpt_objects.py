from typing import TypedDict, Union


# from typing import Optional, Any


class ChatgptResponseUsage(TypedDict):
    """This is the usage object for the ChatGPT API."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# class ChatgptResponseFunctionCall(TypedDict):
#     """This is the function call object for the ChatGPT API."""

#     name: str
#     arguments: str  # Json formatted string with key value arguments


class ChatgptResponseMessage(TypedDict, total=False):
    """This is the message object for the ChatGPT API."""

    role: str
    content: Union[str, None]
    # function_call: Union[ChatgptResponseFunctionCall, None]


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


# class ChatgptRequestFunction(TypedDict):
# """This is the function object for the ChatGPT API."""

# name: str
# description: str
# parameters: dict[str, Any]  # JSON schema object TODO fix


# class ChatgptRequestFunctionCall(TypedDict):
#     """This is the function call object for the ChatGPT API."""

#     name: str
#     arguments: str


class ChatgptRequestMessage(TypedDict, total=False):
    """This is the message object for the ChatGPT API."""

    role: str
    content: Union[str, None]
    name: Union[str, None]
    # function_call: Union[ChatgptRequestFunctionCall, None]


class ChatgptRequest(TypedDict, total=False):
    """This is the request object for chatgpt completions."""

    messages: list[ChatgptRequestMessage]
    # functions: Optional[list[ChatgptRequestFunction]]
    # function_call: Optional[str]


ChatgptMessageHistory = list[ChatgptRequestMessage]
