from typing import Union

from .ephermal_memory import EphermalMemory
from .local_file_memory import LocalFileMemory
from .s3_memory import S3Memory


MemoryProviders = Union[EphermalMemory, LocalFileMemory, S3Memory]

__all__ = [
    "EphermalMemory",
    "LocalFileMemory",
    "S3Memory",
]
