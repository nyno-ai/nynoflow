import pytest
from attrs import define
from pytest_mock import MockerFixture
from transformers import AutoTokenizer

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats._chatgpt._chatgpt_objects import ChatgptResponse
from nynoflow.chats._gpt4all._gpt4all import Gpt4AllProvider
from nynoflow.chats.chat_objects import ChatMessageHistory
from nynoflow.util import logger
from nynoflow.utils.tokenizers.base_tokenizer import BaseTokenizer


chatgpt_response: ChatgptResponse = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Paris",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}


@pytest.fixture(autouse=True)
def mock_openai_chatgpt(mocker: MockerFixture) -> None:
    """Mock the ChatGPT API."""
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)


@define
class Gpt4AllTokenizerOrcaMini3B(BaseTokenizer):
    """Gpt4All tokenizer for the orca mini model."""

    gpt4all_tokenizer = AutoTokenizer.from_pretrained("psmathur/orca_mini_3b")

    def encode(self, text: str) -> list[int]:
        """Encode a string."""
        res: list[int] = self.gpt4all_tokenizer.encode(text)
        return res


class TestChat:
    """Test the Chat class."""

    gpt4all_tokenizer: Gpt4AllTokenizerOrcaMini3B = Gpt4AllTokenizerOrcaMini3B()
    gpt4all_provider = Gpt4AllProvider(
        model_name="orca-mini-3b.ggmlv3.q4_0.bin",
        allow_download=True,
        tokenizer=gpt4all_tokenizer,
        token_limit=400,  # It is actually 1024 but to save some compute time we use 400
    )
    chatgpt_provider = ChatgptProvider(
        organization="myorg",
        api_key="sk-123",
        model="gpt-3.5-turbo-0613",
    )

    def test_mutli_provider(self) -> None:
        """This is a test for the chatgpt function."""
        chat = Chat(providers=[self.chatgpt_provider, self.gpt4all_provider])

        # Make sure no exception is raised when calling the completion function
        chat.completion(prompt="What is the captial of france?", provider_id="chatgpt")
        chat.completion(prompt="What is the captial of italy?", provider_id="gpt4all")
        logger.debug(chat)

    def test_zero_providers(self) -> None:
        """Expect to fail without any providers."""
        with pytest.raises(ValueError):
            Chat(providers=[])

    def test_multiple_providers(self) -> None:
        """Expect to fail with multiple providers with the same id."""
        with pytest.raises(ValueError):
            Chat(
                providers=[
                    ChatgptProvider(
                        provider_id="chatgpt",
                        organization="myorg",
                        api_key="sk-123",
                        model="gpt-3.5-turbo-0613",
                    ),
                    ChatgptProvider(
                        provider_id="chatgpt",
                        organization="myorg",
                        api_key="sk-123",
                        model="gpt-3.5-turbo-0613",
                    ),
                ]
            )

    def test_chat_request_with_invalid_provider_id(self) -> None:
        """Expect to fail with an invalid provider id."""
        with pytest.raises(ValueError):
            Chat(providers=[self.chatgpt_provider]).completion(
                provider_id="invalid",
                prompt="What is the captial of france?",
            )

    def test_chat_request_without_provider_id(self) -> None:
        """Expect to fail with a request to provide a provider id."""
        with pytest.raises(ValueError):
            Chat(providers=[self.chatgpt_provider, self.gpt4all_provider]).completion(
                "What is the captial of france?"
            )

    def test_chat_request_with_valid_provider_id(self) -> None:
        """Expect to succeed with valid provider id or without provider id for one provider."""
        chat = Chat(providers=[self.chatgpt_provider])
        chat.completion("What is the captial of france?")
        chat.completion(
            provider_id="chatgpt",
            prompt="What is the captial of france?",
        )

    def test_message_cutoff(self) -> None:
        """Make sure that old messages are cutoff."""
        chat = Chat(providers=[self.gpt4all_provider])

        # Generate a long list of messages
        messages_before_cutoff = (
            ChatMessageHistory(
                [
                    {
                        "provider_id": "gpt4all",
                        "role": "user",
                        "content": "What is the captial of italy?",
                    },
                    {
                        "provider_id": "gpt4all",
                        "role": "assistant",
                        "content": "Rome. But it is widely known that the capital of Italy is Milan.",
                    },
                ]
            )
            * 100
        )

        chat._message_history.extend(messages_before_cutoff[:])

        # Basically assert that no exception is raised due to the message cutoff
        chat.completion("What is the captical of france?")

    def test_chatgpt_provider_custom_token_limit(self) -> None:
        """Make sure that settings a custom token limit for chatgpt provider works."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    provider_id="chatgpt",
                    model="gpt-3.5-turbo-0613",
                    organization="myorg",
                    api_key="sk-123",
                    token_limit=300,
                )
            ]
        )
        chat.completion("Hello World!")
