# Function Parser Exceptions
class MissingDocstringError(Exception):
    """The function parser is missing a docstring."""

    def __init__(
        self, message: str = "The function parser is missing a docstring."
    ) -> None:
        """Initialize the exception."""
        super().__init__(message)


class MissingDescriptionError(Exception):
    """The function parser is missing a description."""

    def __init__(
        self, message: str = "The function parser is missing a description."
    ) -> None:
        """Initialize the exception."""
        super().__init__(message)


class MissingTypeHintsError(Exception):
    """The function parser is missing type hints."""

    def __init__(
        self,
        message: str = "The fucntion provided is missing type hints for some arguments.",
    ) -> None:
        """Initialize the exception."""
        super().__init__(message)


class InvalidFunctionCallResponseError(Exception):
    """The function call provided by the LLM is invalid."""

    def __init__(
        self, message: str = "The function call provided by the LLM is invalid."
    ) -> None:
        """Initialize the exception."""
        super().__init__(message)


# Chat Exceptions
class ServiceUnavailableError(Exception):
    """The chat recieved a service unavailable error."""


class InvalidResponseError(Exception):
    """The LLM responded with a response that is not valid according to the condition specified."""


class InvalidProvidersError(Exception):
    """The chat recieved invalid providers."""


class ProviderMissingInCompletionError(Exception):
    """The chat must provide a provider id."""

    def __init__(
        self,
        message: str = "You must provide a provider_id to the completion method since"
        "you initialized the Chat class with more then one provider.",
    ) -> None:
        """Initialize the exception."""
        super().__init__(message)


class ProviderNotFoundError(Exception):
    """Unable to find the provider with the given provider id."""

    def __init__(self, provider_id: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"Unable to find the provider with the given provider id ({provider_id})."
        )
