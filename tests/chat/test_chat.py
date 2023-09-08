import pytest
from attrs import define
from gpt4all import GPT4All  # type: ignore
from openai.error import ServiceUnavailableError as OpenaiServiceUnavailableError
from pytest_mock import MockerFixture
from transformers import AutoTokenizer  # type: ignore

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats._chatgpt._chatgpt_objects import ChatgptResponse
from nynoflow.chats._gpt4all._gpt4all import Gpt4AllProvider
from nynoflow.chats.chat_objects import ChatMessageHistory
from nynoflow.exceptions import (
    InvalidProvidersError,
    ProviderMissingInCompletionError,
    ProviderNotFoundError,
    ServiceUnavailableError,
)
from nynoflow.util import logger
from nynoflow.utils.tokenizers.base_tokenizer import BaseTokenizer
from tests.chat.helpers import render_chatgpt_response


chatgpt_response: ChatgptResponse = render_chatgpt_response("Paris")


@pytest.fixture(autouse=True)
def mock_openai_chatgpt(mocker: MockerFixture) -> None:
    """Mock the ChatGPT API."""
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)

    # Avoid downloading the model file
    mocker.patch.object(GPT4All, "__init__", lambda x, y: None)
    # Mock the generated content
    mocker.patch.object(GPT4All, "generate", return_value="Paris")


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
        with pytest.raises(InvalidProvidersError):
            Chat(providers=[])

    def test_multiple_providers(self) -> None:
        """Expect to fail with multiple providers with the same id."""
        with pytest.raises(InvalidProvidersError):
            Chat(
                providers=[
                    ChatgptProvider(
                        provider_id="chatgpt",
                        api_key="sk-123",
                        model="gpt-3.5-turbo-0613",
                    ),
                    ChatgptProvider(
                        provider_id="chatgpt",
                        api_key="sk-123",
                        model="gpt-3.5-turbo-0613",
                    ),
                ]
            )

    def test_chat_request_with_invalid_provider_id(self) -> None:
        """Expect to fail with an invalid provider id."""
        with pytest.raises(ProviderNotFoundError):
            Chat(providers=[self.chatgpt_provider]).completion(
                provider_id="invalid",
                prompt="What is the captial of france?",
            )

    def test_chat_request_without_provider_id(self) -> None:
        """Expect to fail with a request to provide a provider id."""
        with pytest.raises(ProviderMissingInCompletionError):
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
        messages_before_cutoff = ChatMessageHistory(
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
            * 100
        )

        chat._message_history.extend(messages_before_cutoff[:])

        # Basically assert that no exception is raised due to the message cutoff
        chat.completion("What is the captical of france?", token_offset=16)

    def test_chatgpt_provider_custom_token_limit(self) -> None:
        """Make sure that settings a custom token limit for chatgpt provider works."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    provider_id="chatgpt",
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                    token_limit=300,
                )
            ]
        )
        chat.completion("Hello World!")

    def test_service_unavailable_retries_success(self, mocker: MockerFixture) -> None:
        """Test the service unavailable retries succeed after multiple tries."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    provider_id="chatgpt",
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                    retries_on_service_error=5,
                )
            ]
        )

        # Will fail the first 2 times and then return the chatgpt_response
        mocker.patch(
            "openai.ChatCompletion.create",
            side_effect=[
                OpenaiServiceUnavailableError(),
                OpenaiServiceUnavailableError(),
                chatgpt_response,
            ],
        )

        # Will succeed the third time
        assert chat.completion("What is the captical of france?") == "Paris"

    def test_service_unavailable_retries_failure(self, mocker: MockerFixture) -> None:
        """Test the service unavailable error with the chatgpt provider."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                    retries_on_service_error=5,
                )
            ]
        )
        mocker.patch(
            "openai.ChatCompletion.create",
            side_effect=OpenaiServiceUnavailableError(),
        )

        with pytest.raises(ServiceUnavailableError):
            chat.completion("What is the captical of france?")
