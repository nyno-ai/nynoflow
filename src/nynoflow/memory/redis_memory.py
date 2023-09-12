import json
from typing import cast

import cattrs
import redis
from attrs import define, field

from nynoflow.chats import ChatMessage
from nynoflow.memory.base_memory import BaseMemory


@define(kw_only=True)
class RedisMemory(BaseMemory):
    """Redis memory backend."""

    _redis_client: redis.Redis = field()  # type: ignore # redis client type hints requires Generic type falsly

    def load_message_history(self) -> None:
        """Load a chat from backend to memory."""
        # Using cast because redis falsly returns Union[Awaitable[list], list] although it should be without Awaitable.
        # See more: https://github.com/redis/redis-py/issues/2399
        raw_data_list = cast(
            list[bytearray], self._redis_client.lrange(self.chat_id, 0, -1)
        )

        # We reverse the list because redis returns the last message first
        self.message_history = [
            cattrs.structure(json.loads(raw_data.decode("utf-8")), ChatMessage)
            for raw_data in raw_data_list
        ][::-1]

    def _insert_message_backend(self, msg: ChatMessage) -> None:
        """Insert a message into the backend."""
        self._redis_client.lpush(self.chat_id, json.dumps(cattrs.unstructure(msg)))

    def _remove_message_backend(self, msg: ChatMessage) -> None:
        """Remove a message from the backend by letting redis search for the message."""
        self._redis_client.lrem(self.chat_id, 1, json.dumps(cattrs.unstructure(msg)))

    def cleanup(self) -> None:
        """Remove the chat key."""
        self._redis_client.delete(self.chat_id)
