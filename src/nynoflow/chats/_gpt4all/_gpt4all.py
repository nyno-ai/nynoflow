from attrs import define, field
from gpt4all import GPT4All  # type: ignore

from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.tokenizers import BaseTokenizer


@define
class Gpt4AllProvider:
    """LLM Engine for GPT4ALL."""

    model_name: str = field()
    tokenizer: BaseTokenizer = field()
    token_limit: int = field()
    model_path: str = field(default=None)
    allow_download: bool = field(default=False)

    provider_id: str = field(default="gpt4all")
    retries_on_service_error: int = field(default=0)

    _gpt4all_client: GPT4All = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the gpt4all package."""
        self._gpt4all_client = GPT4All(
            model_name=self.model_name,
            model_path=self.model_path,
            allow_download=self.allow_download,
        )

    def completion(self, messages: list[ChatMessage]) -> str:
        """Get a completion from the GPT4ALL API.

        Args:
            messages (list[ChatMessage]): The message history to prompt with.

        Returns:
            The completion.
        """
        res: str = self._gpt4all_client.generate(str(messages))
        return res
