from attrs import define, field
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobClient, BlobServiceClient

from nynoflow.memory.base_file_memory import BaseFileMemory


@define(kw_only=True)
class AzureBlobMemory(BaseFileMemory):
    """Store message history in an Azure Blob Storage."""

    container_name: str = field()
    connection_string: str = field()
    blob_name: str = field()

    @blob_name.default
    def _default_blob_name_factory(self) -> str:
        return f"nynoflow/{str(self.chat_id)}/memory.json"

    _blob_client: BlobClient = field(init=False)

    @_blob_client.default
    def _blob_client_factory(self) -> BlobClient:
        """Create a Blob client."""
        blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        container_client = blob_service_client.get_container_client(self.container_name)
        return container_client.get_blob_client(self.blob_name)

    def _read_memory_file(self) -> str:
        """Read the memory file from Azure Blob. Raise a FileNotFoundError if the file does not exist."""
        try:
            blob_data = self._blob_client.download_blob()
            return blob_data.readall().decode("utf-8")
        except ResourceNotFoundError as err:
            raise FileNotFoundError(
                f"Blob {self.blob_name} does not exist in container {self.container_name}"
            ) from err

    def _write_memory_file(self, content: str) -> None:
        """Write to the Azure Blob. Create the blob if it does not exist."""
        self._blob_client.upload_blob(content, overwrite=True)

    def _remove_memory_file(self) -> None:
        """Remove the memory file from Azure Blob."""
        self._blob_client.delete_blob()
