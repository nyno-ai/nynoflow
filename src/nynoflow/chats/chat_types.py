from typing import Union

from attrs import field
from attrs import frozen


@frozen
class ChatResponse:
    """Generic Chat Message response structure."""

    content: str = field()


@frozen
class ChatRequest:
    """Generic Chat Message request structure."""

    role: str = field()
    content: str = field()
    provider_id: Union[str, None] = field(default=None)
