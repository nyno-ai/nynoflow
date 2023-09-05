import json
from copy import deepcopy
from typing import Any, Literal, Optional, Type, Union
from warnings import warn

from attrs import define, field
from pydantic import BaseModel, Json

from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats._gpt4all._gpt4all import Gpt4AllProvider
from nynoflow.chats.function import Function
from nynoflow.exceptions import (
    InvalidFunctionCallRequestError,
    InvalidFunctionCallResponseError,
    InvalidProvidersError,
    ProviderMissingInCompletionError,
    ProviderNotFoundError,
)
from nynoflow.util import logger
from nynoflow.utils.output_parser import output_parser
from nynoflow.utils.templates.load_template import (
    render_function_call,
    render_functions,
    render_output_formatter,
)

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
        InvalidProvidersError: If multiple providers have the same provider_id or no providers are provided.
    """
    providers = value
    if len(providers) == 0:
        raise InvalidProvidersError("At least one provider must be provided.")

    provider_ids = [p.provider_id for p in providers]
    if len(provider_ids) != len(set(provider_ids)):
        raise InvalidProvidersError(
            "All chat providers must have a unique provider_id."
        )


class FunctionInvocation(BaseModel):
    """Function invocation object."""

    name: str
    arguments: Union[dict[str, Any], Json[dict[str, Any]]]


@define
class Chat:
    """Abstraction class above the different chat providers.

    Args:
        providers (list[_chat_provider]): The chat providers to use.
    """

    providers: list[_chat_provider] = field(validator=_validate_providers)

    _message_history: ChatMessageHistory = field(factory=ChatMessageHistory)

    def completion(
        self,
        prompt: str,
        provider_id: Union[str, None] = None,
        token_offset: int = 16,
        output_format: Optional[Type[BaseModel]] = None,
        functions: Optional[list[Function]] = None,
        function_call: Union[
            dict[Literal["name"], str], Literal["auto"], None
        ] = "auto",
    ) -> Union[str, object]:
        """Chat Completion method for abstracting away the request/resopnse of each provider.

        Raises:
            InvalidFunctionCallRequestError: If no function is found for the function call.

        Args:
            prompt (str): The message to use for the completion.
            provider_id (str, optional): The provider_id to use. Defaults to None (only applicable if there is one provider).
            token_offset (int): The minimum number of tokens to offset for the response. Defaults to 16.
            output_format (Optional[Type[BaseModel]], optional): The output format to use. Defaults to None.
            functions (Optional[list["Function"]], optional): Functions for the LLM for this completion. Defaults to None.
            function_call (Union[dict[Literal["name"], str], str, None]): Provide a function name to force a function call.
                Defaults to "auto" to let the LLM decide (won't affect the pormpt if no functions are provided).
                Provide None to disable function calls.

        Returns:
            str: The completion of the prompt.
        """
        provider = self._get_provider(provider_id)

        formatted_prompt = str(prompt)
        if output_format is not None:
            json_schema = json.dumps(output_format.model_json_schema())
            formatted_prompt = render_output_formatter(prompt, json_schema)

        if functions is not None:
            if function_call == "auto":
                formatted_prompt = render_functions(formatted_prompt, functions)
            elif isinstance(function_call, dict):
                matching_functions = [
                    f for f in functions if f.name == function_call.get("name")
                ]
                if len(matching_functions) == 0:
                    raise InvalidFunctionCallRequestError(
                        f"No function found for function call {function_call}"
                    )
                function = matching_functions[0]
                formatted_prompt = render_function_call(formatted_prompt, function)
            else:
                warn(
                    "You provided functions using the 'functions' arguments but disabled function calls"
                    "by providing 'None' to 'function_call' argument. This will result in the functions not being used."
                )

        self._message_history.append(
            {
                "provider_id": provider.provider_id,
                "content": formatted_prompt,
                "role": "user",
            },
        )
        message_history_after_cutoff = self._cutoff_message_history(
            provider, token_offset
        )
        completion: str = provider.completion(message_history_after_cutoff)
        self._message_history.append(
            {
                "provider_id": provider.provider_id,
                "content": completion,
                "role": "assistant",
            },
        )
        if output_format:
            return output_parser(output_format, completion)

        if functions:
            try:
                # Try parsing the completion as function call
                function_call_completion = FunctionInvocation.model_validate_json(
                    completion
                )
                function_response = self._invoke_function(
                    function_call_completion, functions
                )
                return function_response
            except Exception:
                # No function was called, pass here to return completion later
                logger.debug("No function was called")
                pass

        return completion

    def _invoke_function(
        self, function_call: FunctionInvocation, functions: list[Function]
    ) -> Union[str, object]:
        """Invoke a function and respond with the result.

        Raises:
            InvalidFunctionCallResponseError: If no function is found for the function call.

        Args:
            function_call (FunctionInvocation): The function call to invoke.
            functions (list[Function]): The functions the user provided.

        Returns:
            The response of the function call.
        """
        matching_functions = [f for f in functions if f.name == function_call.name]
        if len(matching_functions) == 0:
            raise InvalidFunctionCallResponseError(
                f"No function found for function call {function_call}"
            )
        func = matching_functions[0]

        response: Union[str, object] = func.invoke(**function_call.arguments)
        return response

    def _cutoff_message_history(
        self, provider: _chat_provider, token_offset: int
    ) -> ChatMessageHistory:
        """Cutoff message history starting from the last message to make sure we have enough tokens for the answer.

        Args:
            provider (_chat_provider): The provider to use.
            token_offset (int): The minimum number of tokens to offset for the response.

        Returns:
            ChatMessageHistory: The cutoff message history.
        """
        message_history = deepcopy(self._message_history)
        while (
            provider.token_limit - provider._num_tokens(message_history) < token_offset
        ):
            message_history.pop(0)

        return message_history

    def _get_provider(self, provider_id: Union[str, None]) -> _chat_provider:
        """Get a provider by its provider_id. If no provider_id is provided and there is only one provider, return it.

        Args:
            provider_id (str): The provider_id to search for.

        Raises:
            ProviderNotFoundError: If no provider is found.
            ProviderMissingInCompletionError: If no provider_id is provided and there are multiple providers.

        Returns:
            The provider with the given provider_id.
        """
        if provider_id is None:
            if len(self.providers) == 1:
                return self.providers[0]
            else:
                raise ProviderMissingInCompletionError()

        providers_found = [p for p in self.providers if p.provider_id == provider_id]

        if len(providers_found) == 0:
            raise ProviderNotFoundError(provider_id)
        # No need to check for multiple providers with the same provider_id as this is already done in _validate_providers
        else:
            return providers_found[0]

    def __str__(self) -> str:
        """Get the string representation of the chat.

        Returns:
            str: The string representation of the chat. Each message is on a new line with the role and the content printed.
        """
        return str(self._message_history)
