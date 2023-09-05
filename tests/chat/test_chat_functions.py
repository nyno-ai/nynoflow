from enum import Enum

import pytest
from pydantic import BaseModel

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats.chat import FunctionInvocation
from nynoflow.chats.function import Function
from nynoflow.exceptions import (
    InvalidFunctionCallRequestError,
    InvalidFunctionCallResponseError,
    MissingDescriptionError,
    MissingDocstringError,
    MissingTypeHintsError,
)
from tests.conftest import ConfigTests


# Example function to help with the testing
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather in a location.

    Args:
        location (str): The city and state.
        unit (str): The unit of the temperature, can be celsius or fahrenheit. Defaults to "celsius".

    Returns:
        str: The weather.
    """
    return f"The weather in {location} is 20 degrees {unit}"


def say_hey(name: str) -> str:
    """Say hello to the user with his name.

    Args:
        name: The name to say hey to.

    Returns:
        str: The message.
    """
    return f"Hello {name}"


class TestChatFunctions:
    """Test the chat function."""

    def test_chatgpt_function(self, config: ConfigTests) -> None:
        """Test the functions of the ChatGPT provider."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        response = chat.completion(
            "What is the weather in boston?",
            functions=[
                Function(
                    name="get_weather",
                    description="Get the weather in a location",
                    func=get_weather,
                    parameters={
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state.",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ],
        )
        assert isinstance(response, str)
        assert response.lower() == "The weather in boston is 20 degrees celsius".lower()

    def test_chatgpt_function_from_function(self, config: ConfigTests) -> None:
        """Test the functions of the ChatGPT provider."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        response = chat.completion(
            "What is the weather in boston?",
            functions=[Function.from_function(get_weather)],
        )
        assert isinstance(response, str)
        assert response.lower() == "The weather in boston is 20 degrees celsius".lower()


class TestChatFunctionParser:
    """Test the function parser of the Function class."""

    def test_function_parser_simple_parameters(self) -> None:
        """Expect the function parser to succeed with the function name and description."""

        def hello_world(a: str, b: int) -> None:
            """This is the start of the description.

            This is some more description.

            Args:
                a: The argument without type hint.
                b: The argument with a type hint.
            """

        function = Function.from_function(hello_world)
        assert function.name == "hello_world"
        assert (
            function.description
            == "This is the start of the description. This is some more description."
        )

    def test_function_parser_invalid_docstring_parameter_warning(self) -> None:
        """Expect warning when the function parser is provided with a docstring without parameter description."""

        def func(a: str) -> None:
            """This is an invalid docstring - no parameters are defined."""

        with pytest.warns():
            Function.from_function(func)

    def test_function_parser_no_docstring(self) -> None:
        """Expect MissingDocstringError when the function parser is provided without a docstring."""

        def func() -> None:
            pass  # pragma: no cover # Ignored because the function is not called

        with pytest.raises(MissingDocstringError):
            Function.from_function(func)

    def test_function_parser_empty_docstring(self) -> None:
        """Expect MissingDescriptionError when the function parser is provided without an empty docstring."""

        def func() -> None:
            """"""  # noqa: D419

        with pytest.raises(MissingDescriptionError):
            Function.from_function(func)

    def test_function_parser_no_type_hints(self) -> None:
        """Expect MissingTypeHintsError when the function parser is provided with an argument without type hint."""

        def func(a) -> None:  # type: ignore
            """This is a valid docstring.

            Args:
                a: The argument without type hint.
            """

        with pytest.raises(MissingTypeHintsError):
            Function.from_function(func)

    def test_function_parser_default_value(self) -> None:
        """Expect valid default values for default variables."""

        def func(a: str, b: str = "hello!") -> None:
            """This is a valid docstring.

            Args:
                a: The argument without type hint.
                b: The argument with a default value.
            """

        schema = Function.from_function(func)
        assert schema.parameters["properties"]["b"]["default"] == "hello!"

    def test_function_parse_complex(self) -> None:
        """Expect function parser to succeed with complex types, arguments and docstring combination."""

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        class Person(BaseModel):
            name: str
            age: int
            favorite_color: Color

        def func(
            a: int,
            p: Person,
            d: dict[str, str],
            c: Color = Color.RED,
        ) -> None:
            """This is a valid docstring.

            Args:
                a: The argument without type hint.
                p: The argument with a complex type.
                c: The argument with a default value of a complex type.
                d: The argument with a default value of a complex type.
            """

        result = Function.from_function(func)
        assert result.parameters["properties"]["c"]["default"] == "red"


class TestFunctionCall:
    """Test the function_call feature."""

    def test_auto_function_call(self, config: ConfigTests) -> None:
        """Test an auto function call."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        response = chat.completion(
            "Hey my name is john",
            functions=[Function.from_function(say_hey)],
            function_call={"name": "say_hey"},
        )
        assert isinstance(response, str)
        assert response.lower() == "hello john"

    def test_invalid_function_call_name_in_request(self, config: ConfigTests) -> None:
        """Test an invalid function call name in the user request."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        with pytest.raises(InvalidFunctionCallRequestError):
            chat.completion(
                "Hey my name is john",
                functions=[Function.from_function(say_hey)],
                function_call={"name": "INVALID_NAME_SHOULD_BE_SAY_HEY"},
            )

    def test_warning_for_disabling_function_call_with_functions(
        self, config: ConfigTests
    ) -> None:
        """Make sure the user is warned when he provides functions but disables function call."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        with pytest.warns():
            chat.completion(
                "Hey my name is john",
                functions=[Function.from_function(say_hey)],
                function_call=None,
            )


class TestFunctionInvoke:
    """Test the responde of the LLM with the function invocation parameters."""

    def test_valid_function_call(self, config: ConfigTests) -> None:
        """Test a valid function call."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        response = chat.completion(
            "Hey my name is john",
            functions=[Function.from_function(say_hey)],
            function_call={"name": "say_hey"},
        )
        assert isinstance(response, str)
        assert response.lower() == "hello john"

    def test_invalid_function_name(self, config: ConfigTests) -> None:
        """Test an invalid function call name in the LLM response."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        with pytest.raises(InvalidFunctionCallResponseError):
            chat._invoke_function(
                FunctionInvocation(
                    name="INVALID_FUNCTION_NAME",
                    arguments={"name": "valid_argument"},
                ),
                functions=[Function.from_function(say_hey)],
            )

    def test_missing_arguments(self, config: ConfigTests) -> None:
        """Test a function call with missing arguments."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        with pytest.raises(InvalidFunctionCallResponseError):
            chat._invoke_function(
                FunctionInvocation(name="say_hey", arguments={}),
                functions=[Function.from_function(say_hey)],
            )

    def test_invalid_arguments(self, config: ConfigTests) -> None:
        """Test a function call with invalid arguments."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        with pytest.raises(InvalidFunctionCallResponseError):
            chat._invoke_function(
                FunctionInvocation(name="say_hey", arguments={"name": 123}),
                functions=[Function.from_function(say_hey)],
            )

    def test_function_error(self, config: ConfigTests) -> None:
        """Test a function call with invalid arguments."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        class MyError(Exception):
            pass

        def func(a: str) -> None:
            """My function."""
            raise MyError("This is a test error")

        with pytest.raises(MyError):
            chat._invoke_function(
                FunctionInvocation(name="func", arguments={"a": "test"}),
                functions=[Function.from_function(func)],
            )
