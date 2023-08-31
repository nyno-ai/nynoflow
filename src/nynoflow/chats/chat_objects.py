from typing import TypedDict


class ChatMessage(TypedDict):
    """This is the message object for the chat class."""

    provider_id: str
    content: str
    role: str


class ChatMessageHistory(list[ChatMessage]):
    """Representation of the chat message history for the chat class."""

    def __str__(self) -> str:
        """Get a string representation of the message history."""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self])
