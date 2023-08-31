from attrs import define, field
from gpt4all import GPT4All

from nynoflow.chats.chat_objects import ChatMessageHistory

# from ._gpt4all_objects import GPT4AllModel
from nynoflow.util import logger

# from transformers import AutoTokenizer
from nynoflow.utils.tokenizers.base_tokenizer import BaseTokenizer


@define
class Gpt4AllProvider:
    """LLM Engine for GPT4ALL."""

    model_name: str = field()
    tokenizer: BaseTokenizer = field()
    model_path: str = field(default=None)
    allow_download: bool = field(default=False)

    provider_id: str = field(default="gpt4all")

    _gpt4all_client: GPT4All = field(init=False)
    _token_limit: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the gpt4all package."""
        logger.debug("Configuring Gpt4All client")
        self._gpt4all_client = GPT4All(
            model_name=self.model_name,
            model_path=self.model_path,
            allow_download=self.allow_download,
        )

        # logger.debug("Configure tokenizer for Gpt4All client")
        # gpt4all_models: list[GPT4AllModel] = self._gpt4all_client.list_models()
        # model_matches: list[GPT4AllModel] = [
        #     m for m in gpt4all_models if m["filename"] == self.model_name
        # ]
        # if len(model_matches) == 0:
        #     raise ValueError(f"Model {self.model_name} not found.")

        # if len(model_matches) > 1:
        #     raise ValueError(f"Multiple models found for model name {self.model_name}.")

        # model: GPT4AllModel = model_matches[0]
        # # Convert  'https://huggingface.co/nomic-ai/gpt4all-falcon-ggml/resolve/main/ggml-model-gpt4all-falcon-q4_0.bin'
        # # to 'nomic-ai/gpt4all-falcom-ggml'
        # huggnigface_repo_name = "/".join(model["url"].split("/")[3:5])
        # self.tokenizer = AutoTokenizer.from_pretrained(huggnigface_repo_name)
        # self._token_limit = self._tokenizer.max_model_input_sizes[
        #     "max_model_input_sizes"
        # ].values()[0]

    def _num_tokens(self, messages: ChatMessageHistory) -> int:
        """Get the number of tokens in a message list for this provider and tokenizer.

        Args:
            messages: List of messages

        Returns:
            The number of tokens in the message list.
        """
        return self.tokenizer.token_count(self._convert_message_history(messages))

    def _convert_message_history(self, message_history: ChatMessageHistory) -> str:
        """Convert the message history to a GPT4All format message history.

        Args:
            message_history (ChatMessageHistory): The message history to convert.

        Returns:
            str: The converted message history.
        """
        return "\n".join([f"{msg.role}: {msg.content}\n" for msg in message_history])

    def completion(self, prompt: str) -> str:
        """Get a completion from the GPT4ALL API.

        Args:
            prompt (str): The prompt to use for the completion.

        Returns:
            The completion.
        """
        res: str = self._gpt4all_client.generate(prompt)
        return res
