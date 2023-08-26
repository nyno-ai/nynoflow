from attrs import define
from attrs import field

from nynoflow.util import logger

from ._chatgpt import ChatgptProvider
from ._gpt4all import Gpt4AllProvider
from .chat_types import ChatRequest
from .chat_types import ChatResponse


@define
class Chat:
    """Abstraction class above the different chat providers.

    Args:
        providers (list[ChatgptProvider | Gpt4AllProvider]): The chat providers to use.
    """

    providers: list[ChatgptProvider | Gpt4AllProvider] = field()

    _message_history: list[str] = []
    _tokens_left: int = 1000
    _token_consumption: int = 0

    def completion(self, message: ChatRequest) -> ChatResponse:
        """Chat Completion method for abstracting away the request/resopnse of each provider.

        Raises:
            ValueError: If no compatible provider is found or multiple compatible providers are found.

        Args:
            message (ChatRequest): The message to use for the completion.

        Returns:
            ChatResponse object with the result.
        """
        logger.debug(f"Completion invoked. Message: {message}")

        if len(self.providers) == 1:
            provider = self.providers[0]
        else:
            compatible_providers = [
                p for p in self.providers if p.provider_id == message.provider_id
            ]
            if len(compatible_providers) == 0:
                raise ValueError(
                    f"No compatible provider found for provider id {message.provider_id}"
                )
            elif len(compatible_providers) > 1:
                raise ValueError(
                    f"Multiple compatible providers found for provider id {message.provider_id}. Provider ids must be unique."
                )
            else:
                provider = compatible_providers[0]

        res: ChatResponse = provider.completion(message)
        return res
