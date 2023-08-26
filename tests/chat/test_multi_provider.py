import pytest
from pytest_mock import MockerFixture

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt import ChatgptProvider
from nynoflow.chats._chatgpt import ChatgptRequest
from nynoflow.chats._chatgpt import ChatgptRequestMessage
from nynoflow.chats._chatgpt import ChatgptResponse
from nynoflow.chats._chatgpt import ChatgptResponseChoice
from nynoflow.chats._chatgpt import ChatgptResponseMessage
from nynoflow.chats._chatgpt import ChatgptResponseUsage
from nynoflow.chats._gpt4all import Gpt4AllProvider
from nynoflow.chats.chat_types import ChatRequest


chatgpt_response_message = ChatgptResponseMessage(
    role="assistant",
    content="Paris",
)

chatgpt_response_choice = ChatgptResponseChoice(
    index=0,
    message=chatgpt_response_message,
    finish_reason="stop",
)

chatgpt_response = ChatgptResponse(
    id="chatcmpl-123",
    object="chat.completion",
    created=1677652288,
    model="gpt-3.5-turbo-0613",
    choices=[chatgpt_response_choice],
    usage=ChatgptResponseUsage(prompt_tokens=9, completion_tokens=12, total_tokens=21),
)

chatgpt_request = ChatgptRequest(
    model="gpt-3.5-turbo",
    messages=[
        ChatgptRequestMessage(role="user", content="What is the captial of france?")
    ],
)


def test_mutli_provider(mocker: MockerFixture) -> None:
    """This is a test for the chatgpt function.

    Args:
        mocker: The mocker object.
    """
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)
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
        ChatRequest(
            provider_id="chatgpt", role="user", content="What is the captial of france?"
        )
    )
    res2 = chat.completion(
        ChatRequest(
            provider_id="gpt4all", role="user", content="What is the captial of italy?"
        )
    )
    assert res1 and res2


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


def test_chat_request_with_invalid_provider_id(mocker: MockerFixture) -> None:
    """Expect to fail with an invalid provider id."""
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)
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
            ChatRequest(
                provider_id="invalid",
                role="user",
                content="What is the captial of france?",
            )
        )


def test_chat_request_without_provider_id(mocker: MockerFixture) -> None:
    """Expect to fail with a request to provide a provider id."""
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)
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
            ChatRequest(
                role="user",
                content="What is the captial of france?",
            )
        )


def test_chat_request_with_valid_provider_id(mocker: MockerFixture) -> None:
    """Expect to succeed with valid provider id."""
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)
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
        ChatRequest(
            provider_id="chatgpt",
            role="user",
            content="What is the captial of france?",
        )
    )
