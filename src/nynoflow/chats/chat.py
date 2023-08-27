from typing import Any
from typing import Union

from attrs import define
from attrs import field

from nynoflow.util import logger

from ._chatgpt import ChatgptProvider
from ._gpt4all import Gpt4AllProvider
from .chat_types import ChatRequest
from .chat_types import ChatResponse


_chat_provider = Union[ChatgptProvider, Gpt4AllProvider]


def _validate_providers(
    instance: "Chat",
    # Have to use Any because if we use the correct type (Attribute[list[_chat_provider]])
    # we get 'type' object is not subscriptable. It isn't too important because we don't use
    # the attribute variable in the function and it's an internal one. See more here:
    # https://github.com/python-attrs/attrs/issues/524
    attribute: Any,
    value: list[_chat_provider],
) -> None:
    """Validate that at least one provider exists and that all providers have a unique provider_id.

    Args:
        instance: The Chat object.
        attribute: The attribute to validate.
        value: The value of the attribute.

    Raises:
        ValueError: If multiple providers have the same provider_id.
    """
    providers = value
    if len(providers) == 0:
        raise ValueError("At least one provider must be provided.")

    provider_ids = [p.provider_id for p in providers]
    if len(provider_ids) != len(set(provider_ids)):
        raise ValueError("All chat providers must have a unique provider_id.")


@define
class Chat:
    """Abstraction class above the different chat providers.

    Args:
        providers (list[ChatgptProvider | Gpt4AllProvider]): The chat providers to use.
    """

    providers: list[_chat_provider] = field(validator=_validate_providers)

    _message_history: list[str] = []
    _tokens_left: int = 1000
    _token_consumption: int = 0

    def _get_provider(self, provider_id: Union[str, None]) -> _chat_provider:
        """Get a provider by its provider_id.

        Args:
            provider_id (str): The provider_id to search for.

        Raises:
            ValueError: If no provider is found.

        Returns:
            The provider with the given provider_id.
        """
        if provider_id is None:
            if len(self.providers) == 1:
                return self.providers[0]
            else:
                raise ValueError(
                    "No provider_id provided and multiple providers exist."
                )

        providers_found = [p for p in self.providers if p.provider_id == provider_id]

        if len(providers_found) == 0:
            raise ValueError(f"No provider found for provider id {provider_id}")
        # No need to check for multiple providers with the same provider_id as this is already done in _validate_providers
        else:
            return providers_found[0]

    def completion(self, message: ChatRequest) -> ChatResponse:
        """Chat Completion method for abstracting away the request/resopnse of each provider.

        Args:
            message (ChatRequest): The message to use for the completion.

        Returns:
            ChatResponse object with the result.
        """
        logger.debug(f"Completion invoked. Message: {message}")
        provider = self._get_provider(message.provider_id)
        res: ChatResponse = provider.completion(message)
        return res
