from typing import Any, Literal, TypeVar, Union
from uuid import uuid4

from attrs import define, field
from pydantic import BaseModel, Json


# class ChatMessage(BaseModel):
@define
class ChatMessage:
    """This is the message object for the chat class.

    Attributes:
        provider_id (str): The id of the message in the provider.
        content (str): The content of the message.
        role (str): The role of the message. Can be "user", "assistant", "system" or "function".
        temporary (bool): Whether the message is temporary or not. Used for example for fixing the LLM response.
    """

    provider_id: str = field()
    content: str = field()
    role: Literal["user", "assistant", "system", "function"] = field()
    temporary: bool = field(default=False)

    # _id: str = Field(default_factory=lambda: str(uuid4()))
    _id: str = field(factory=lambda: str(uuid4()))

    def __str__(self) -> str:
        """Get a string representation of the message."""
        return f"{self.role} ({self.provider_id}): {self.content}"


class FunctionInvocation(BaseModel):
    """Function invocation object."""

    name: str
    arguments: Union[dict[str, Any], Json[dict[str, Any]]]


OutputFormatterType = TypeVar("OutputFormatterType", bound=BaseModel)
AutoFixerType = TypeVar("AutoFixerType", bound=Any)
