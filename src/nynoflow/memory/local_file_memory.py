import os

from attrs import define, field

from nynoflow.memory.base_file_memory import BaseFileMemory


@define(kw_only=True)
class LocalFileMemory(BaseFileMemory):
    """Store message history in a local file."""

    file_path: str = field()

    @file_path.default
    def _default_file_path(self) -> str:
        return os.path.join(".", ".nynoflow", str(self.chat_id), "memory.json")

    def _read_memory_file(self) -> str:
        """Read the memory file. Raise a FileNotFoundError if the file does not exist."""
        with open(self.file_path) as f:
            return f.read()

    def _write_memory_file(self, content: str) -> None:
        """Write to the memory file. Create the file if it does not exist."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            f.write(content)

    def _remove_memory_file(self) -> None:
        """Remove the memory file."""
        os.remove(self.file_path)
