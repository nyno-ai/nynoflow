import openai
import pytest

from nynoflow.chats._chatgpt._chatgpt_objects import ChatgptMessageHistory
from nynoflow.utils.tokenizers.openai_tokenizer import ChatgptTiktokenTokenizer
from tests.conftest import ConfigTests


class TestOpenaiTokenizer:
    """Test the openai tokenizer."""

    models = [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4",
    ]

    messages = ChatgptMessageHistory(
        [
            {
                "role": "system",
                "content": "You are an assistant",
            },
            {
                "role": "user",
                "name": "example_user",
                "content": "How are you?",
            },
            {
                "role": "assistant",
                "content": "Very good thank you. How can I help you?",
            },
        ]
    )

    def test_token_count(self, config: ConfigTests) -> None:
        """Test tokenizer for models without function support."""
        for model in self.models:
            response = openai.ChatCompletion.create(
                api_key=config["OPENAI_API_KEY"],
                model=model,
                messages=self.messages,
                temperature=0,
                max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
            )
            usage_tokens = response["usage"]["prompt_tokens"]

            tokenizer = ChatgptTiktokenTokenizer(model)
            calculated_tokens = tokenizer.token_count(self.messages)
            assert usage_tokens == calculated_tokens

    def test_invalid_model(self, config: ConfigTests) -> None:
        """Make sure the tokenizer raises an exception for invalid model names."""
        with pytest.warns():
            ChatgptTiktokenTokenizer("invalid-model-name")
