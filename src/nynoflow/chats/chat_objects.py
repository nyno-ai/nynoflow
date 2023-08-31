# from attrs import define, field
from typing import TypedDict


# @define
# class ChatMessageRequest:
#     """This is the message object for the chat class."""

#     provider_id: str = field()
#     content: str = field()
#     role: str = field(default="user")
#     # name: Union[str, None] = field(default=None)
#     # functions: Union[dict[str, str], None] = field(default=None)

#     def __str__(self) -> str:
#         """Get the string representation of the message.

#         Returns:
#             str: The string representation of the message.
#         """
#         return f"""{self.role}: {self.content}"""


# @define
# class ChatMessageResponse:
#     """This is the response message object for the chat class."""

#     # Save historically the provider that responded, not used in the chat
#     provider_id: str = field()
#     content: str = field()
#     role: str = field(default="assistant")
#     # function_call: Union[dict[str, str], None] = field(default=None)


class ChatMessage(TypedDict):
    """This is the message object for the chat class."""

    provider_id: str
    content: str
    role: str


# ChatMessageHistory = list[Union[ChatMessageRequest, ChatMessageResponse]]
ChatMessageHistory = list[ChatMessage]
