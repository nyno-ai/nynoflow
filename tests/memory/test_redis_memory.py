from typing import Generator
from uuid import uuid4

import fakeredis
import pytest

from nynoflow.memory import MemoryProviders, RedisMemory
from tests.conftest import ConfigTests
from tests.memory.base_memory_tests import BaseMemoryTest


@pytest.fixture(scope="function")
def memory(config: ConfigTests) -> Generator[RedisMemory, None, None]:
    """Return a unique memory client on each call."""
    fake_redis_client = fakeredis.FakeStrictRedis()
    yield RedisMemory(
        chat_id=str(uuid4()), redis_client=fake_redis_client, persist=False
    )


class TestRedisMemory(BaseMemoryTest):
    """Test memory implementations."""

    def is_backend_memory_exists(self, memory: MemoryProviders) -> bool:
        """Check if the memory key exists in Redis."""
        assert isinstance(memory, RedisMemory)
        return memory._redis_client.exists(memory.chat_id) == 1
