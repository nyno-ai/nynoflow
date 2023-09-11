import json
from typing import Callable, Type

import cattrs
from attrs import define, field
from sqlalchemy import JSON, Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from nynoflow.chats import ChatMessage
from nynoflow.memory.base_memory import BaseMemory


Base = declarative_base()


def create_message_record_table(table_name: str) -> Type[Base]:
    """Used to insert the table_name dynamically for the MessageRecord class."""

    class MessageRecord(Base):
        """Message record in the SQL database."""

        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        id = Column(String, primary_key=True)
        chat_id = Column(String, nullable=False)
        message = Column(JSON, nullable=False)

    return MessageRecord


@define(kw_only=True)
class SQLAlchemyMemory(BaseMemory):
    """SQLAlchemy memory backend."""

    db_url: str = field()
    table_name: str = field(default="message_history")

    MessageRecord = field(init=False)

    @MessageRecord.default
    def _message_record_factory(self) -> Type[Base]:
        """Create a message record table with the given table name."""
        return create_message_record_table(self.table_name)

    Session = field(init=False)

    @Session.default
    def _session_factory(self) -> Callable[[], sessionmaker]:
        """Create a session factory."""
        engine = create_engine(self.db_url)
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)

    def load_message_history(self) -> None:
        """Load a chat from backend to memory."""
        session = self.Session()
        records = (
            session.query(self.MessageRecord).filter_by(chat_id=self.chat_id).all()
        )
        session.close()

        self.message_history = [
            cattrs.structure(json.loads(record.message), ChatMessage)
            for record in records
        ]

    def _insert_message_backend(self, msg: ChatMessage) -> None:
        """Insert the message as a new table row."""
        session = self.Session()
        record = self.MessageRecord(
            chat_id=self.chat_id,
            message=json.dumps(cattrs.unstructure(msg)),
            id=msg._id,
        )
        session.add(record)
        session.commit()
        session.close()

    def _remove_message_backend(self, msg: ChatMessage) -> None:
        """Remove the message row."""
        session = self.Session()
        session.query(self.MessageRecord).filter_by(
            chat_id=self.chat_id, id=msg._id
        ).delete()
        session.commit()
        session.close()

    def cleanup(self) -> None:
        """Delete all chat messages filtered by chat_id."""
        session = self.Session()
        session.query(self.MessageRecord).filter_by(chat_id=self.chat_id).delete()
        session.commit()
        session.close()
