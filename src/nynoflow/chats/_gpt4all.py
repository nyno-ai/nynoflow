from attrs import asdict
from attrs import define
from attrs import field
from attrs import frozen
from gpt4all import GPT4All

from nynoflow.util import logger

from .chat_types import ChatRequest
from .chat_types import ChatResponse


@frozen
class Gpt4AllRequest:
    """This is the request object for the GPT4All API."""

    prompt: str = field()


@frozen  # Frozen because configuration should be immutable
class Gpt4AllConfig:
    """GPT4All Chat Engine specific configuration.

    Args:
        endpoint: The endpoint to use for the GPT4All API.
    """


@define
class Gpt4AllProvider:
    """LLM Engine for GPT4ALL."""

    model_name: str = field()
    model_path: str = field(default=None)
    allow_download: bool = field(default=False)

    provider_id: str = field(default="chatgpt")

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

    def completion(self, request: ChatRequest) -> ChatResponse:
        """Get a completion from the GPT4ALL API.

        Args:
            request: The prompt to use for the completion.

        Returns:
            The completion.
        """
        logger.debug(f"Generating completion for {request}")
        gpt4all_request = self._convert_request(request)
        logger.debug(f"Converted request: {gpt4all_request}")
        gpt4all_response: str = self._gpt4all_client.generate(**asdict(gpt4all_request))
        logger.debug(f"Generated completion: {gpt4all_response}")
        response = self._convert_response(gpt4all_response)
        return response

    def _convert_request(self, request: ChatRequest) -> Gpt4AllRequest:
        """Convert a chat request to a GPT4All request."""
        logger.debug(f"Converting request: {request}")
        x = Gpt4AllRequest(prompt=request.content)
        return x

    def _convert_response(self, response: str) -> ChatResponse:
        """Convert a chat response to a generic response."""
        logger.debug(f"Converting response: {response}")
        return ChatResponse(content=response)
