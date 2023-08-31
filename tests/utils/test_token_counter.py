import openai
import pytest

from nynoflow.chats._chatgpt._chatgpt_objects import (  # ChatgptRequestFunction,
    ChatgptMessageHistory,
)
from nynoflow.util import logger
from nynoflow.utils.tokenizers.openai_tokenizer import ChatgptTiktokenTokenizer
from tests.conftest import ConfigTests


class TestOpenaiTokenizer:
    """Test the openai tokenizer."""

    MODELS_WITH_FUNCTION_SUPPORT = [
        "gpt-3.5-turbo-0613",
        # "gpt-3.5-turbo",
        # "gpt-4-0314",
        # "gpt-4-0613",
        # "gpt-4",
    ]

    MODELS_WITHOUT_FUNCTION_SUPPORT = [
        "gpt-3.5-turbo-0301",
    ]

    # FUNCTIONS: list[ChatgptRequestFunction] = [
    #     {
    #         "name": "get_current_weather",
    #         "description": "Get the current weather in a given location",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "location": {
    #                     "type": "string",
    #                     "description": "The city and state, e.g. San Francisco, CA",
    #                 },
    #                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #             },
    #             "required": ["location"],
    #         },
    #     },
    #     {
    #         "name": "my_random_function",
    #         "description": "Just a function without any parameters",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #         },
    #     },
    # ]

    MESSAGES_WITHOUT_FUNCTIONS: ChatgptMessageHistory = [
        {
            "role": "system",
            "content": "You are an assistant",
        },
        {
            "role": "user",
            "name": "example_user",
            "content": "How are you?",
        },
    ]
    MESSAGES_WITH_FUNCTIONS: ChatgptMessageHistory = [
        {
            "role": "system",
            "content": "You are an assistant\n",
        },
        {
            "role": "user",
            "name": "example_user",
            "content": "What is the weather in boston?",
        },
        # {
        #     "role": "assistant",
        #     "content": None,
        #     "function_call": {
        #         "name": "get_current_weather",
        #         "arguments": "{\n  \"location\": \"Boston, MA\"\n}",
        #     },
        # },
        # {
        #     "role": "function",
        #     "name": "get_current_weather",
        #     "content": '{"location": "boston", "temperature": "72", "unit": "celsius", "forecast": ["sunny", "windy"]}',
        # },
        {
            "role": "assistant",
            "content": "The weather is 72 celsius degrees.",
        },
        {
            "role": "user",
            "content": "thanks! how about the weather in new york?",
        },
        # {
        #     "role": "assistant",
        #     "content": None,
        #     "function_call": {
        #         "name": "get_current_weather",
        #         "arguments": '{"location": "new york"}',
        #     },
        # },
        # {
        #     "role": "function",
        #     "name": "get_current_weather",
        #     "content": '{"location": "new york", "temperature": "13", "unit": "celsius", "forecast": ["sunny", "windy"]}',
        # },
        # {
        #     "role": "assistant",
        #     "content": "The weather is 13 celsius degrees.",
        # },
    ]

    def test_with_functions(self, config: ConfigTests) -> None:
        """Test tokenizer for models with function support."""
        for model in self.MODELS_WITH_FUNCTION_SUPPORT:
            response = openai.ChatCompletion.create(
                api_key=config["OPENAI_API_KEY"],
                model=model,
                # functions=self.FUNCTIONS,
                messages=self.MESSAGES_WITH_FUNCTIONS,
                temperature=0,
                max_tokens=1000,  # we're only counting input tokens here, so let's not waste tokens on the output
            )
            logger.debug(response)
            usage_tokens = response["usage"]["prompt_tokens"]

            tokenizer = ChatgptTiktokenTokenizer(model)
            calculated_tokens = tokenizer.token_count(self.MESSAGES_WITH_FUNCTIONS)
            assert usage_tokens == calculated_tokens

    def test_without_functions(self, config: ConfigTests) -> None:
        """Test tokenizer for models without function support."""
        for model in self.MODELS_WITHOUT_FUNCTION_SUPPORT:
            response = openai.ChatCompletion.create(
                api_key=config["OPENAI_API_KEY"],
                model=model,
                messages=self.MESSAGES_WITHOUT_FUNCTIONS,
                temperature=0,
                max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
            )
            usage_tokens = response["usage"]["prompt_tokens"]

            tokenizer = ChatgptTiktokenTokenizer(model)
            calculated_tokens = tokenizer.token_count(self.MESSAGES_WITHOUT_FUNCTIONS)
            assert usage_tokens == calculated_tokens

    def test_invalid_model(self, config: ConfigTests) -> None:
        """Make sure the tokenizer raises an exception for invalid model names."""
        with pytest.warns():
            ChatgptTiktokenTokenizer("invalid-model-name")
