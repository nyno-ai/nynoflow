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


# Function Call Exceptions
class InvalidFunctionCallRequestError(Exception):
    """The function call provided by the user is invalid."""

    def __init__(
        self, message: str = "The function call provided by the user is invalid."
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


# Output Parser Exceptions
class InvalidOutputError(Exception):
    """The output parser recieved invalid output."""

    def __init__(
        self, message: str = "The output parser recieved invalid output."
    ) -> None:
        """Initialize the exception."""
        super().__init__(message)
