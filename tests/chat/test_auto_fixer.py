import pytest
from pytest_mock import MockerFixture

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats.chat_objects import ChatMessageHistory
from nynoflow.exceptions import InvalidResponseError
from tests.chat.helpers import render_chatgpt_response


@pytest.fixture(autouse=True)
def mock_openai_chatgpt(mocker: MockerFixture) -> None:
    """Mock the ChatGPT API."""
    mocker.patch(
        "openai.ChatCompletion.create", return_value=render_chatgpt_response("Paris")
    )


class TestChatAutoFixer:
    """Test the Chat class."""

    chatgpt_provider = ChatgptProvider(
        api_key="sk-123",
        model="gpt-3.5-turbo-0613",
    )

    def test_auto_fixer(self, mocker: MockerFixture) -> None:
        """Test the auto fixer. We can't patch the completion method because it is a read only attribute."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                )
            ]
        )
        result = "Paris"

        call_count = 0

        def my_auto_fixer(response: str) -> str:
            nonlocal call_count
            call_count += 1

            if call_count <= 2:  # Throw exceptions for the first two calls
                raise InvalidResponseError("Please do this fix and that fix")

            return result

        llm_response, function_output = chat.completion_with_auto_fixer(
            "What is the captical of france?",
            auto_fixer=my_auto_fixer,
            auto_fixer_retries=4,
        )

        assert function_output == result
        assert chat._message_history[0]["content"] == "What is the captical of france?"
        assert chat._message_history[1]["content"] == result

    def test_history_cleaner(self) -> None:
        """Make sure the history cleaner works as intended."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                )
            ]
        )

        chat._message_history = ChatMessageHistory(
            [
                {
                    "provider_id": "chatgpt",
                    "role": "user",
                    "content": "What is the captial of italy?",
                },
                # Deleting Start
                {
                    "provider_id": "chatgpt",
                    "role": "assistant",
                    "content": "Paris",
                },
                {
                    "provider_id": "chatgpt",
                    "role": "user",
                    "content": "This is incorrect. What is the captical of italy?",
                },
                {
                    "provider_id": "chatgpt",
                    "role": "assistant",
                    "content": "Paris",
                },
                {
                    "provider_id": "chatgpt",
                    "role": "user",
                    "content": "This is incorrect. What is the captical of italy?",
                },
                # Deletion End
                {
                    "provider_id": "chatgpt",
                    "role": "assistant",
                    "content": "Rome.",
                },
            ]
        )
        attempts = 3
        chat._clean_auto_fixer_failed_attempts(
            failed_attempts=attempts - 1
        )  # Calculate the failed attempts

        assert len(chat._message_history) == 2
        assert chat._message_history[0]["content"] == "What is the captial of italy?"
        assert chat._message_history[1]["content"] == "Rome."
