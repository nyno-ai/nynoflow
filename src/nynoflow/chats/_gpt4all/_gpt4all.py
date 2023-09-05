from attrs import define, field
from gpt4all import GPT4All  # type: ignore

from nynoflow.chats.chat_objects import ChatMessageHistory
from nynoflow.utils.tokenizers.base_tokenizer import BaseTokenizer


@define
class Gpt4AllProvider:
    """LLM Engine for GPT4ALL."""

    model_name: str = field()
    tokenizer: BaseTokenizer = field()
    token_limit: int = field()
    model_path: str = field(default=None)
    allow_download: bool = field(default=False)

    provider_id: str = field(default="gpt4all")

    _gpt4all_client: GPT4All = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the gpt4all package."""
        self._gpt4all_client = GPT4All(
            model_name=self.model_name,
            model_path=self.model_path,
            allow_download=self.allow_download,
        )

    def _num_tokens(self, messages: ChatMessageHistory) -> int:
        """Get the number of tokens in a message list for this provider and tokenizer.

        Args:
            messages: List of messages

        Returns:
            The number of tokens in the message list.
        """
        return self.tokenizer.token_count(str(messages))

    def completion(self, messages: ChatMessageHistory) -> str:
        """Get a completion from the GPT4ALL API.

        Args:
            messages (ChatMessageHistory): The message history to prompt with.

        Returns:
            The completion.
        """
        res: str = self._gpt4all_client.generate(str(messages))
        return res
