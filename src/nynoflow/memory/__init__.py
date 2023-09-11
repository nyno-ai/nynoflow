from typing import Union

from .ephermal_memory import EphermalMemory
from .gcp_memory import GcpBlobMemory
from .local_file_memory import LocalFileMemory
from .redis_memory import RedisMemory
from .s3_memory import S3Memory
from .sqlalchemy_memory import SQLAlchemyMemory


MemoryProviders = Union[
    EphermalMemory,
    LocalFileMemory,
    S3Memory,
    GcpBlobMemory,
    RedisMemory,
    SQLAlchemyMemory,
]

__all__ = [
    "EphermalMemory",
    "LocalFileMemory",
    "S3Memory",
    "GcpBlobMemory",
    "RedisMemory",
    "SQLAlchemyMemory",
]
