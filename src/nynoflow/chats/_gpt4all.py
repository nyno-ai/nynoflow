from attrs import define, field
from gpt4all import GPT4All

from nynoflow.util import logger


@define
class Gpt4AllProvider:
    """LLM Engine for GPT4ALL."""

    model_name: str = field()
    model_path: str = field(default=None)
    allow_download: bool = field(default=False)

    provider_id: str = field(default="gpt4all")

    _gpt4all_client: GPT4All = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the gpt4all package."""
        logger.debug("Configuring Gpt4All.")
        self._gpt4all_client = GPT4All(
            model_name=self.model_name,
            model_path=self.model_path,
            allow_download=self.allow_download,
        )
        logger.debug("Gpt4All configured.")

    def completion(self, prompt: str) -> str:
        """Get a completion from the GPT4ALL API.

        Args:
            prompt (str): The prompt to use for the completion.

        Returns:
            The completion.
        """
        res: str = self._gpt4all_client.generate(prompt)
        return res
