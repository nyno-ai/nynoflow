import json
from typing import Any, Callable, Type, Union
from uuid import uuid4
from warnings import warn

from attrs import Factory, define, field

from nynoflow.chats import (
    AutoFixerType,
    ChatMessage,
    ChatProvider,
    FunctionInvocation,
    OutputFormatterType,
)
from nynoflow.exceptions import (
    InvalidFunctionCallResponseError,
    InvalidProvidersError,
    InvalidResponseError,
    ProviderMissingInCompletionError,
    ProviderNotFoundError,
    ServiceUnavailableError,
)
from nynoflow.function import Function
from nynoflow.memory import EphermalMemory, MemoryProviders
from nynoflow.templates import (
    render_optional_functions,
    render_output_formatter,
    render_required_functions,
)


@define
class Flow:
    """Abstraction class above the different chat providers.

    Args:
        providers (list[ChatProvider]): The chat providers to use.
    """

    providers: list[ChatProvider] = field()
    chat_id: str = field(factory=lambda: str(uuid4))
    memory_provider: MemoryProviders = Factory(
        lambda self: EphermalMemory(chat_id=self.chat_id), takes_self=True
    )

    @providers.validator
    def _validate_providers(
        self,
        # Have to use Any because if we use the correct type (Attribute[list[ChatProvider]])
        # we get 'type' object is not subscriptable. It isn't too important because the attribute
        # variable in the function is unused and it's an internal function. See more here:
        # https://github.com/python-attrs/attrs/issues/524
        attribute: Any,
        value: list[ChatProvider],
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

    def completion_with_auto_fixer(
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
        provider = self._get_provider(provider_id)

        initial_user_message = ChatMessage(
            provider_id=provider.provider_id,
            content=current_prompt,
            role="user",
        )

        self.memory_provider.insert_message(initial_user_message)

        for attempt in range(1, auto_fixer_retries + 2):
            message_history_after_cutoff = (
                self.memory_provider.get_message_history_upto_token_limit(
                    token_limit=provider.token_limit - token_offset,
                    tokenizer=provider.tokenizer,
                )
            )
            response = self._prompt_provider_with_retry(
                provider=provider,
                messages=message_history_after_cutoff,
                token_offset=token_offset,
            )
            assistant_message = ChatMessage(
                provider_id=provider.provider_id,
                content=response,
                role="assistant",
            )
            try:
                result = auto_fixer(response)
                self.memory_provider.insert_message(assistant_message)
                self.memory_provider.clean_temporary_message_history()
                return response, result
            except InvalidResponseError as err:
                last_exception: InvalidResponseError = err
                warn(
                    f"Failed to fix response {response} due to error {err}. Retrying. Attempt number {attempt}"
                )
                current_prompt = str(err)
                assistant_message.temporary = True
                self.memory_provider.insert_message(assistant_message)

        raise InvalidResponseError from last_exception.__cause__

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
            provider_id=provider.provider_id,
            content=prompt,
            role="user",
        )
        self.memory_provider.insert_message(user_message)
        message_history_after_cutoff = (
            self.memory_provider.get_message_history_upto_token_limit(
                token_limit=provider.token_limit - token_offset,
                tokenizer=provider.tokenizer,
            )
        )

        try:
            completion: str = self._prompt_provider_with_retry(
                provider=provider,
                messages=message_history_after_cutoff,
                token_offset=token_offset,
            )
            assistant_message = ChatMessage(
                provider_id=provider.provider_id, content=completion, role="assistant"
            )
            self.memory_provider.insert_message(assistant_message)
            return completion

        except Exception as err:
            # Cleanup user message in case unexpected error occurs
            self.memory_provider.remove_message(user_message)
            raise err

    def _prompt_provider_with_retry(
        self,
        provider: ChatProvider,
        messages: list[ChatMessage],
        token_offset: int,
    ) -> str:
        """Prompt a provider with retries for service unavailable errors.

        Raises:
            ServiceUnavailableError: If the provider is unavailable after the retries.

        Args:
            provider (ChatProvider): The provider to use.
            messages (list[ChatMessage]): The message history to use for the prompt.
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
                    f"due to {err}. Attempt number {attempt}"
                )
                last_exception = err

        raise ServiceUnavailableError from last_exception.__cause__

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

    def _get_provider(self, provider_id: Union[str, None]) -> ChatProvider:
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
        return "\n".join([str(msg) for msg in self.memory_provider.message_history])
