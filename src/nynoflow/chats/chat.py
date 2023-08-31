from typing import Any, Union

from attrs import define, field

from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats._gpt4all._gpt4all import Gpt4AllProvider

from .chat_objects import ChatMessageHistory


_chat_provider = Union[ChatgptProvider, Gpt4AllProvider]


def _validate_providers(
    instance: "Chat",
    # Have to use Any because if we use the correct type (Attribute[list[_chat_provider]])
    # we get 'type' object is not subscriptable. It isn't too important because the attribute
    # variable in the function is unused and it's an internal function. See more here:
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
        providers (list[_chat_provider]): The chat providers to use.
    """

    providers: list[_chat_provider] = field(validator=_validate_providers)

    _message_history = ChatMessageHistory()

    def completion(self, prompt: str, provider_id: Union[str, None] = None) -> str:
        """Chat Completion method for abstracting away the request/resopnse of each provider.

        Args:
            prompt (str): The message to use for the completion.
            provider_id (str, optional): The provider_id to use. Defaults to None (only applicable if there is one provider).

        Returns:
            str: The completion of the prompt.
        """
        provider = self._get_provider(provider_id)
        message_history_after_cutoff = self._cutoff_message_history(provider)
        completion = provider.completion(message_history_after_cutoff)
        self._message_history.extend(
            [
                {
                    "provider_id": provider.provider_id,
                    "content": prompt,
                    "role": "user",
                },
                {
                    "provider_id": provider.provider_id,
                    "content": completion,
                    "role": "assistant",
                },
            ]
        )
        return completion

    def _cutoff_message_history(self, provider) -> ChatMessageHistory:
        """Cutoff message history starting from the last message to make sure we have enough tokens for the answer.

        Args:
            provider: The provider to use.

        Returns:
            ChatMessageHistory: The cutoff message history.
        """
        message_history = self._message_history[:]
        while (
            provider.token_limit - provider._num_tokens(message_history) < 256
        ):  # TODO change from 256
            message_history.pop(0)

        return message_history

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

    def __str__(self) -> str:
        """Get the string representation of the chat.

        Returns:
            str: The string representation of the chat. Each message is on a new line with the role and the content printed.
        """
        return str(self._message_history)
