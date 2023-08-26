import openai
from attrs import asdict
from attrs import define
from attrs import field
from attrs import frozen

from nynoflow.util import logger

from .chat_types import ChatRequest
from .chat_types import ChatResponse


@frozen
class ChatgptResponseUsage:
    """This is the usage object for the ChatGPT API."""

    prompt_tokens: int = field()
    completion_tokens: int = field()
    total_tokens: int = field()


@frozen
class ChatgptResponseFunctionCall:
    """This is the function call object for the ChatGPT API."""

    name: str = field()
    arguments: str = field()


@frozen
class ChatgptResponseMessage:
    """This is the message object for the ChatGPT API."""

    role: str = field()
    content: str | None = field(default=None)
    function_call: ChatgptResponseFunctionCall | None = field(default=None)


@frozen
class ChatgptResponseChoice:
    """This is the choice object for the ChatGPT API."""

    index: int = field()
    message: ChatgptResponseMessage = field()
    finish_reason: str = field()


@frozen
class ChatgptResponse:
    """This is the object returned by the OpenAI API for a chat completion.

    Parameters are taken from here: https://platform.openai.com/docs/api-reference/chat/object
    """

    id: str = field()
    object: str = field()
    created: int = field()
    model: str = field()
    choices: list[ChatgptResponseChoice] = field()
    usage: ChatgptResponseUsage = field()


@frozen
class ChatgptRequestFunction:
    """This is the function object for the ChatGPT API."""

    name: str = field()
    description: str = field()
    parameters: object = field()


@frozen
class ChatgptRequestFunctionCall:
    """This is the function call object for the ChatGPT API."""

    name: str = field()
    arguments: str = field()


@frozen
class ChatgptRequestMessage:
    """This is the message object for the ChatGPT API."""

    role: str = field()
    content: str = field()
    name: str | None = field(default=None)
    function_call: ChatgptRequestFunctionCall | None = field(default=None)


@frozen
class ChatgptRequest:
    """This is the request object for chatgpt completions.

    Parameters are taken from herer: https://platform.openai.com/docs/api-reference/chat/create?lang=python
    """

    model: str = field()
    messages: list[ChatgptRequestMessage] = field()
    functions: list[ChatgptRequestFunction] | None = field(default=None)
    function_call: str | None = field(default=None)
    temperature: float | None = field(default=None)
    top_p: float | None = field(default=None)
    n: int | None = field(default=None)
    stream = False  # Streams not supported in this library yet
    stop: str | list[str] | None = field(default=None)
    max_tokens: int | None = field(default=None)
    presence_penalty: float | None = field(default=None)
    frequency_penalty: float | None = field(default=None)
    logit_bias: dict[str, float] | None = field(default=None)
    user: str | None = field(default=None)


@define
class ChatgptProvider:
    """LLM Engine for OpenAI ChatGPT."""

    organization: str = field()
    api_key: str = field()
    model: str = field()
    provider_id: str = field(default="gpt4all")

    def __attrs_post_init__(self) -> None:
        """Configure the openai package with auth and configurations."""
        logger.debug("Configuring ChatGPT Provider")
        openai.api_key = self.api_key
        openai.organization = self.organization

    def completion(self, request: ChatRequest) -> ChatResponse:
        """Get a completion from the OpenAI API.

        Args:
            request: The prompt to use for the completion.

        Returns:
            The completion.
        """
        req = self._convert_request(request)
        res: ChatgptResponse = openai.ChatCompletion.create(**asdict(req))
        return self._convert_response(res)

    def _convert_request(self, request: ChatRequest) -> ChatgptRequest:
        """Convert the generic ChatRequest to ChatgptRequest to be used by openai api."""
        return ChatgptRequest(
            model=self.model,
            messages=[
                ChatgptRequestMessage(
                    role=request.role,
                    content=request.content,
                )
            ],
        )

    def _convert_response(self, response: ChatgptResponse) -> ChatResponse:
        """Convert the generic ChatResponse to ChatgptResponse to be used by openai api."""
        return ChatResponse(content=response.choices[0].message.content or "")
