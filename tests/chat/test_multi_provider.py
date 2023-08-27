import pytest

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt import ChatgptProvider
from nynoflow.chats._gpt4all import Gpt4AllProvider
from nynoflow.util import logger


def test_mutli_provider() -> None:
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


def test_zero_providers() -> None:
    """Expect to fail without any providers."""
    with pytest.raises(ValueError):
        Chat(providers=[])


def test_multiple_providers() -> None:
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


def test_chat_request_with_invalid_provider_id() -> None:
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


def test_chat_request_without_provider_id() -> None:
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


def test_chat_request_with_valid_provider_id() -> None:
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
