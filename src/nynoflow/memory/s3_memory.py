from typing import Optional

import boto3
from attrs import define, field
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client

from nynoflow.memory.base_file_memory import BaseFileMemory


@define(kw_only=True)
class S3Memory(BaseFileMemory):
    """Store message history in an AWS S3 bucket."""

    bucket_name: str = field()
    region_name: Optional[str] = field(default=None)
    key: str = field()

    @key.default
    def _default_key_factory(self) -> str:
        return f"nynoflow/{str(self.chat_id)}/memory.json"

    _s3_client: S3Client = field(init=False)

    @_s3_client.default
    def _s3_client_factory(self) -> S3Client:
        """Create an S3 client, optionally specifying a region_name if it's set."""
        if self.region_name is None:
            return boto3.client("s3")
        return boto3.client("s3", region_name=self.region_name)

    def _read_memory_file(self) -> str:
        """Read the memory file from S3. Raise a FileNotFoundError if the file does not exist."""
        try:
            obj = self._s3_client.get_object(Bucket=self.bucket_name, Key=self.key)
            return obj["Body"].read().decode("utf-8")
        except ClientError as err:
            if err.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(
                    f"S3 key {self.key} does not exist in bucket {self.bucket_name}"
                ) from err
            else:
                raise err

    def _write_memory_file(self, content: str) -> None:
        """Write to the S3 bucket. Create the file if it does not exist."""
        self._s3_client.put_object(Body=content, Bucket=self.bucket_name, Key=self.key)

    def _remove_memory_file(self) -> None:
        """Remove the memory file from S3."""
        self._s3_client.delete_object(Bucket=self.bucket_name, Key=self.key)
