from typing import Union

from .ephermal_memory import EphermalMemory
from .gcp_memory import GcpBlobMemory
from .local_file_memory import LocalFileMemory
from .s3_memory import S3Memory


MemoryProviders = Union[EphermalMemory, LocalFileMemory, S3Memory, GcpBlobMemory]

__all__ = [
    "EphermalMemory",
    "LocalFileMemory",
    "S3Memory",
    "GcpBlobMemory",
]
