import pytest
from pytest_mock import MockerFixture

from nynoflow.chats._chatgpt import (
    ChatgptResponse,
    ChatgptResponseChoice,
    ChatgptResponseMessage,
    ChatgptResponseUsage,
)


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


@pytest.fixture(autouse=True)
def mock_openai_chatgpt(mocker: MockerFixture) -> None:
    """Mock the ChatGPT API."""
    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)
