from typing import Union

from .ephermal_memory import EphermalMemory
from .local_file_memory import LocalFileMemory


MemoryProviders = Union[EphermalMemory, LocalFileMemory]

__all__ = [
    "EphermalMemory",
    "LocalFileMemory",
]
