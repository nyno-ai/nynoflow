import pytest
from pytest_mock import MockerFixture

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt import ChatgptProvider, ChatgptResponse
from nynoflow.chats._gpt4all import Gpt4AllProvider
from nynoflow.util import logger


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


class TestChat:
    """Test the Chat class."""

    def test_mutli_provider(self) -> None:
        """This is a test for the chatgpt function."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    organization="myorg",
                    api_key="sk-123",
                    model="gpt-3.5-turbo",
                ),
                Gpt4AllProvider(
                    model_name="orca-mini-3b.ggmlv3.q4_0.bin", allow_download=True
                ),
            ]
        )

        res1 = chat.completion(
            prompt="What is the captial of france?", provider_id="chatgpt"
        )

        res2 = chat.completion(
            prompt="What is the captial of italy?", provider_id="gpt4all"
        )

        logger.debug(res1)
        logger.debug(res2)

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
                        model="gpt-3.5-turbo",
                    ),
                    ChatgptProvider(
                        provider_id="chatgpt",
                        organization="myorg",
                        api_key="sk-123",
                        model="gpt-3.5-turbo",
                    ),
                ]
            )

    def test_chat_request_with_invalid_provider_id(self) -> None:
        """Expect to fail with an invalid provider id."""
        with pytest.raises(ValueError):
            Chat(
                providers=[
                    ChatgptProvider(
                        provider_id="chatgpt",
                        organization="myorg",
                        api_key="sk-123",
                        model="gpt-3.5-turbo",
                    ),
                ]
            ).completion(
                provider_id="invalid",
                prompt="What is the captial of france?",
            )

    def test_chat_request_without_provider_id(self) -> None:
        """Expect to fail with a request to provide a provider id."""
        with pytest.raises(ValueError):
            Chat(
                providers=[
                    ChatgptProvider(
                        organization="myorg",
                        api_key="sk-123",
                        model="gpt-3.5-turbo",
                    ),
                    Gpt4AllProvider(
                        model_name="orca-mini-3b.ggmlv3.q4_0.bin", allow_download=True
                    ),
                ]
            ).completion(
                prompt="What is the captial of france?",
            )

    def test_chat_request_with_valid_provider_id(self) -> None:
        """Expect to succeed with valid provider id or without provider id for one provider."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    provider_id="chatgpt",
                    organization="myorg",
                    api_key="sk-123",
                    model="gpt-3.5-turbo",
                ),
            ]
        )
        chat.completion(
            provider_id="chatgpt",
            prompt="What is the captial of france?",
        )

        chat.completion(
            prompt="What is the captial of france?",
        )
