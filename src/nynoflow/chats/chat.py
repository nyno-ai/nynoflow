import json
from copy import deepcopy
from typing import Any, Callable, Type, TypeVar, Union
from warnings import warn

from attrs import define, field
from pydantic import BaseModel, Json

from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats._gpt4all._gpt4all import Gpt4AllProvider
from nynoflow.chats.chat_objects import ChatMessage, ChatMessageHistory
from nynoflow.chats.function import Function
from nynoflow.exceptions import (
    InvalidFunctionCallResponseError,
    InvalidProvidersError,
    InvalidResponseError,
    ProviderMissingInCompletionError,
    ProviderNotFoundError,
    ServiceUnavailableError,
)
from nynoflow.utils.templates.render_templates import (
    render_optional_functions,
    render_output_formatter,
    render_required_functions,
)


OutputFormatterType = TypeVar("OutputFormatterType", bound=BaseModel)
AutoFixerType = TypeVar("AutoFixerType", bound=Any)
ChatProviderType = Union[ChatgptProvider, Gpt4AllProvider]


class FunctionInvocation(BaseModel):
    """Function invocation object."""

    name: str
    arguments: Union[dict[str, Any], Json[dict[str, Any]]]


@define
class Chat:
    """Abstraction class above the different chat providers.

    Args:
        providers (list[ChatProviderType]): The chat providers to use.
    """

    providers: list[ChatProviderType] = field()

    @providers.validator
    def _validate_providers(
        self,
        # Have to use Any because if we use the correct type (Attribute[list[ChatProviderType]])
        # we get 'type' object is not subscriptable. It isn't too important because the attribute
        # variable in the function is unused and it's an internal function. See more here:
        # https://github.com/python-attrs/attrs/issues/524
        attribute: Any,
        value: list[ChatProviderType],
    ) -> None:
        """Validate that at least one provider exists and that all providers have a unique provider_id.

        Args:
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

    _message_history: ChatMessageHistory = field(factory=ChatMessageHistory)

    def completion_with_functions(
        self,
        prompt: str,
        functions: list[Function[Any]],
        require_function_call: bool = False,
        auto_fix_retries: int = 0,
        provider_id: Union[str, None] = None,
        token_offset: int = 16,
    ) -> Any:
        """Renders the prompt in a template designed to get a function invocation, and return the function result.

        This method is used to be sure to get a function invocation for one of the functions provided. If you want
        to allow the LLM to optionally invoke a function, use completion_with_optional_functions instead.

        Args:
            prompt (str): The message to use for the completion.
            functions (list[Function]): Functions for the LLM for this completion.
            require_function_call (bool): Whether the LLM must invoke a function or not. Defaults to False.
            auto_fix_retries (int): The number of times to retry in case the LLM respond with invalid data structure. Defaults to 0.
            provider_id (str, optional): The provider_id to use. Defaults to None (only applicable if there is one provider).
            token_offset (int): The minimum number of tokens to offset for the response. Defaults to 16.

        Returns:
            Any: The response from the invoked function.
        """
        if require_function_call:
            formatted_prompt = render_required_functions(prompt, functions)
        else:
            formatted_prompt = render_optional_functions(prompt, functions)

        def function_auto_fixer(response: str) -> Any:
            """Auto fix function invocation.

            Will try to parse the LLM response as function invocation. If it fails, and function invocation is
            not required, it will return the plain response because it means it's a regular completion.
            If it fails but function invocation is required, it will raise an InvalidResponseError.
            After parsing the function invocation, wether if it's required or not, it will try to invoke
            the function. If the function invocation fails with InvalidFunctionCallResponseError it means
            it tried to invoke a function but provided invalid parameters, so it'll raise the exception to retry.

            Raises:
                InvalidResponseError: If the response is not valid.

            Args:
                response (str): The response from the LLM.

            Returns:
                Any: The response from the invoked function.
            """
            try:
                function_invocation = FunctionInvocation.model_validate_json(response)
            except ValueError as err:
                if require_function_call is False:
                    return response

                raise InvalidResponseError(
                    "Please provide a function invocation that adheres to the format specified in the previous request"
                ) from err

            try:
                return self._invoke_function(function_invocation, functions)
            except InvalidFunctionCallResponseError as err:
                raise InvalidResponseError() from err

        response, function_return_value = self.completion_with_auto_fixer(
            prompt=formatted_prompt,
            auto_fixer=function_auto_fixer,
            auto_fixer_retries=auto_fix_retries,
            provider_id=provider_id,
            token_offset=token_offset,
        )

        return function_return_value

    def completion_with_output_formatter(
        self,
        prompt: str,
        output_format: Type[OutputFormatterType],
        auto_fix_retries: int = 0,
        provider_id: Union[str, None] = None,
        token_offset: int = 16,
    ) -> OutputFormatterType:
        """Renders the prompt in a template designed to get a response in the output format, and returning the parsed output.

        Args:
            prompt (str): The message to use for the completion.
            output_format (Type[BaseModel]): The output format to use.
            auto_fix_retries (int): The number of times to retry in case the LLM respond with invalid data structure. Defaults to 0.
            provider_id (str, optional): The provider_id to use. Defaults to None (only applicable if there is one provider).
            token_offset (int): The minimum number of tokens to offset for the response. Defaults to 16.

        Returns:
            OutputFormatterType: The parsed output of the completion.
        """

        def output_formatter_auto_fixer(response: str) -> OutputFormatterType:
            """Create a validation function for the auto fixer.

            Raises:
                InvalidResponseError: If the response is not valid.

            Args:
                response (str): The LLM response.

            Returns:
                OutputFormatterType: The parsed output of the completion.
            """
            try:
                return output_format.model_validate_json(response)
            except ValueError as err:
                raise InvalidResponseError("Response is invalid") from err

        json_schema_format = json.dumps(output_format.model_json_schema())
        formatted_prompt = render_output_formatter(prompt, json_schema_format)
        response, obj = self.completion_with_auto_fixer(
            prompt=formatted_prompt,
            auto_fixer=output_formatter_auto_fixer,
            auto_fixer_retries=auto_fix_retries,
            provider_id=provider_id,
            token_offset=token_offset,
        )

        return obj

    def completion_with_auto_fixer(  # type: ignore # MyPy false positive about not always returning a value from the loop
        self,
        prompt: str,
        auto_fixer: Callable[[str], AutoFixerType],
        auto_fixer_retries: int = 0,
        provider_id: Union[str, None] = None,
        token_offset: int = 16,
    ) -> tuple[str, AutoFixerType]:
        """Chat Completion method for abstracting away the request/resopnse of each provider.

        Raises:
            InvalidResponseError: In case the retries are exceeded and the response is still not valid.

        Args:
            prompt (str): The message to use for the completion.
            auto_fixer (Callable[[str], Any]): Function to verify the response. Raise an InvalidResponseError if the
                response is not valid. Return the response or a parsed versioned of if it is valid.
            auto_fixer_retries (int): The number of times to retry the auto fixer. Defaults to 0.
            provider_id (str, optional): The provider_id to use. Defaults to None (only applicable if there is one provider).
            token_offset (int): The minimum number of tokens to offset for the response. Defaults to 16.

        Returns:
            tuple[str, AutoFixerType]: A tuple containing the response and the result of the auto fixer function.
                0: The final completion from the LLM.
                1: The return value of the auto fixer function.
        """
        current_prompt = str(prompt)

        for attempt in range(1, auto_fixer_retries + 2):
            response = self.completion(
                prompt=current_prompt,
                provider_id=provider_id,
                token_offset=token_offset,
            )
            try:
                result = auto_fixer(response)
                self._clean_auto_fixer_failed_attempts(failed_attempts=attempt - 1)
                return response, result
            except InvalidResponseError as err:
                last_exception: InvalidResponseError = err
                warn(
                    f"Failed to fix response {response} due to error {err}. Retrying. Attempt number {attempt}"
                )
                current_prompt = str(err)

        raise InvalidResponseError(last_exception)

    def _clean_auto_fixer_failed_attempts(self, failed_attempts: int) -> None:
        """Clean the message history from the failed attempts of the auto fixer.

        The algorithm is to keep the last message which is the valid response from the
        model, and the first message the user prompted before all the attempts.

        Args:
            failed_attempts (int): The number of failed attempts before succeeding.
        """
        last_message = self._message_history[-1]

        # failed attempts * 2 because each attempt is 2 messages, one for user and one for assistant.
        messages_until_user_prompt = self._message_history[
            : len(self._message_history) - failed_attempts * 2 - 1
        ]

        self._message_history = ChatMessageHistory(
            messages_until_user_prompt + [last_message]
        )

    def completion(
        self,
        prompt: str,
        provider_id: Union[str, None] = None,
        token_offset: int = 16,
    ) -> str:
        """Chat Completion method for abstracting away the request/resopnse of each provider.

        Raises:
            Exception: Proxies the underlying exception if anything goes wrong.

        Args:
            prompt (str): The message to use for the completion.
            provider_id (str, optional): The provider_id to use. Defaults to None (only applicable if there is one provider).
            token_offset (int): The minimum number of tokens to offset for the response. Defaults to 16.

        Returns:
            str: The completion of the prompt.
        """
        provider = self._get_provider(provider_id)

        user_message = ChatMessage(
            {
                "provider_id": provider.provider_id,
                "content": prompt,
                "role": "user",
            }
        )
        self._message_history.append(user_message)
        message_history_after_cutoff = self._cutoff_message_history(
            provider, token_offset
        )
        try:
            completion: str = self._prompt_provider_with_retry(
                provider=provider,
                messages=message_history_after_cutoff,
                token_offset=token_offset,
            )
            assistant_message = ChatMessage(
                {
                    "provider_id": provider.provider_id,
                    "content": completion,
                    "role": "assistant",
                }
            )
            self._message_history.append(assistant_message)
            return completion

        except Exception as err:
            # Cleanup user message in case unexpected error occurs
            self._message_history.remove(user_message)
            raise err

    def _prompt_provider_with_retry(  # type: ignore # MyPy false positive about not always returning a value from the loop
        self,
        provider: ChatProviderType,
        messages: ChatMessageHistory,
        token_offset: int,
    ) -> str:
        """Prompt a provider with retries for service unavailable errors.

        Raises:
            ServiceUnavailableError: If the provider is unavailable after the retries.

        Args:
            provider (ChatProviderType): The provider to use.
            messages (ChatMessageHistory): The message history to use for the prompt.
            token_offset (int): The minimum number of tokens to offset for the response.

        Returns:
            str: The completion of the prompt.
        """
        for attempt in range(1, provider.retries_on_service_error + 2):
            try:
                return provider.completion(messages)
            except ServiceUnavailableError as err:
                warn(
                    f"Failed to get completion from provider {provider.provider_id}"
                    f"due to error {err}. Attempt number {attempt}"
                )
                last_exception = err

        raise ServiceUnavailableError(last_exception)

    def _invoke_function(
        self, function_invocation: FunctionInvocation, functions: list[Function[Any]]
    ) -> Any:
        """Invoke a function and respond with the result.

        Raises:
            InvalidFunctionCallResponseError: If no function is found for the function call.

        Args:
            function_invocation (FunctionInvocation): The function call to invoke.
            functions (list[Function[Any]]): The functions the user provided.

        Returns:
            The response of the function call.
        """
        matching_functions = [
            f for f in functions if f.name == function_invocation.name
        ]
        if len(matching_functions) == 0:
            raise InvalidFunctionCallResponseError(
                f"No function found for function call {function_invocation}"
            )
        func = matching_functions[0]

        response = func.invoke(**function_invocation.arguments)
        return response

    def _cutoff_message_history(
        self, provider: ChatProviderType, token_offset: int
    ) -> ChatMessageHistory:
        """Cutoff message history starting from the last message to make sure we have enough tokens for the answer.

        Args:
            provider (ChatProviderType): The provider to use.
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

    def _get_provider(self, provider_id: Union[str, None]) -> ChatProviderType:
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
