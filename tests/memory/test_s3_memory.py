from typing import Generator
from uuid import uuid4

import boto3
import pytest
from botocore.exceptions import ClientError

from nynoflow.memory import MemoryProviders, S3Memory
from tests.conftest import ConfigTests
from tests.memory.base_file_memory_tests import BaseFileMemoryTest


@pytest.fixture(scope="function")
def memory(config: ConfigTests) -> Generator[S3Memory, None, None]:
    """Return a unique memory client on each call."""
    yield S3Memory(
        chat_id=str(uuid4()), bucket_name=config["AWS_S3_BUCKET_NAME"], persist=False
    )


class TestS3Memory(BaseFileMemoryTest):
    """Test memory implementations."""

    def is_memory_file_exists(self, memory: MemoryProviders) -> bool:
        """Check if the memory file exists."""
        # Needed for type checking.
        assert isinstance(memory, S3Memory)
        s3 = boto3.client("s3")

        try:
            s3.head_object(Bucket=memory.bucket_name, Key=memory.key)
            return True
        except ClientError as err:
            if err.response["Error"]["Code"] == "404":
                return False

            # Unreachable exception because the client will fail before if no
            # credentials, but needed for type checking.
            raise err  # pragma: no cover

    def test_s3_bucket_doesnt_exist(self, config: ConfigTests) -> None:
        """Test that the S3 client is created with the correct region."""
        with pytest.raises(ClientError):
            S3Memory(
                chat_id=str(uuid4()),
                bucket_name="just-invalid-bucket-name-that-doesnt-exist",
                persist=False,
            )
