from typing import Union

from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats._gpt4all._gpt4all import Gpt4AllProvider

from .chat_objects import (
    AutoFixerType,
    ChatMessage,
    FunctionInvocation,
    OutputFormatterType,
)


ChatProvider = Union[ChatgptProvider, Gpt4AllProvider]

__all__ = [
    "ChatgptProvider",
    "Gpt4AllProvider",
    "ChatProvider",
    "ChatMessage",
    "list[ChatMessage]",
    "FunctionInvocation",
    "OutputFormatterType",
    "AutoFixerType",
]
