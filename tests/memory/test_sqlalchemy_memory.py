from typing import Generator
from uuid import uuid4

import pytest

# from sqlalchemy import create_engine
from nynoflow.memory import MemoryProviders, SQLAlchemyMemory

# from nynoflow.memory.sqlalchemy_memory import MessageRecord
from tests.memory.base_memory_tests import BaseMemoryTest


@pytest.fixture(scope="function")
def memory() -> Generator[SQLAlchemyMemory, None, None]:
    """Return a unique memory client on each call. Using sqlite in memory for easiest testing."""
    db_url = "sqlite:///:memory:"
    yield SQLAlchemyMemory(chat_id=str(uuid4()), db_url=db_url, persist=False)


class TestSQLAlchemyMemory(BaseMemoryTest):
    """Test memory implementations."""

    def is_backend_memory_exists(self, memory: MemoryProviders) -> bool:
        """Return True if the memory backend exists."""
        assert isinstance(memory, SQLAlchemyMemory)
        session = memory.Session()
        count = (
            session.query(memory.MessageRecord)
            .filter_by(chat_id=memory.chat_id)
            .count()
        )
        session.close()

        is_exists: bool = count > 0
        return is_exists
