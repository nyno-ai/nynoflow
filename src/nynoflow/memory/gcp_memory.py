from attrs import define, field
from google.api_core.exceptions import NotFound
from google.cloud.storage import Blob, Client  # type: ignore

from nynoflow.memory.base_file_memory import BaseFileMemory


@define(kw_only=True)
class GcpBlobMemory(BaseFileMemory):
    """Store message history in a GCP bucket."""

    bucket_name: str = field()

    key: str = field()

    @key.default
    def _default_key_factory(self) -> str:
        return f"nynoflow/{str(self.chat_id)}/memory.json"

    gcp_client: Client = field(factory=Client)

    blob: Blob = field(init=False)

    @blob.default
    def _default_blob_factory(self) -> Blob:
        return self.gcp_client.bucket(self.bucket_name).blob(self.key)

    def _read_memory_file(self) -> str:
        """Read the memory file from GCP.

        Raises:
            FileNotFoundError: if the file does not exist.

        Returns:
            str: the content of the file.
        """
        try:
            content: str = self.blob.download_as_text()
            return content
        except NotFound as err:
            raise FileNotFoundError("Blob not found") from err

    def _write_memory_file(self, content: str) -> None:
        """Write to the GCP bucket. Create the file if it does not exist."""
        self.blob.upload_from_string(content)

    def _remove_memory_file(self) -> None:
        """Remove the memory file from GCP."""
        self.blob.delete()
