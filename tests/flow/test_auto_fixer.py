import pytest
from pytest_mock import MockerFixture

from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.exceptions import InvalidResponseError
from nynoflow.flow import Flow
from tests.helpers import render_chatgpt_response


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
        flow = Flow(
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

        llm_response, function_output = flow.completion_with_auto_fixer(
            "What is the captical of france?",
            auto_fixer=my_auto_fixer,
            auto_fixer_retries=4,
        )

        assert function_output == result
        assert (
            flow.memory_provider.message_history[0].content
            == "What is the captical of france?"
        )
        assert flow.memory_provider.message_history[1].content == result

    def test_history_cleaner(self) -> None:
        """Make sure the history cleaner works as intended."""
        flow = Flow(
            providers=[
                ChatgptProvider(
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                )
            ]
        )

        flow.memory_provider.message_history = list[ChatMessage](
            [
                ChatMessage(
                    provider_id="chatgpt",
                    role="user",
                    content="What is the captial of italy?",
                ),
                # Deleting Start
                ChatMessage(
                    provider_id="chatgpt",
                    role="assistant",
                    content="Paris",
                    temporary=True,
                ),
                ChatMessage(
                    provider_id="chatgpt",
                    role="user",
                    content="This is incorrect. What is the captical of italy?",
                    temporary=True,
                ),
                ChatMessage(
                    provider_id="chatgpt",
                    role="assistant",
                    content="Paris",
                    temporary=True,
                ),
                ChatMessage(
                    provider_id="chatgpt",
                    role="user",
                    content="This is incorrect. What is the captical of italy?",
                    temporary=True,
                ),
                # Deletion End
                ChatMessage(
                    provider_id="chatgpt",
                    role="assistant",
                    content="Rome.",
                ),
            ]
        )
        flow.memory_provider.clean_temporary_message_history()

        assert len(flow.memory_provider.message_history) == 2
        assert (
            flow.memory_provider.message_history[0].content
            == "What is the captial of italy?"
        )
        assert flow.memory_provider.message_history[1].content == "Rome."

    def test_auto_fixer_failure(self) -> None:
        """Test that the auto fixer fails after too many failures."""

        def my_invalid_auto_fixer(response: str) -> str:
            raise InvalidResponseError("Please do this fix and that fix")

        flow = Flow(
            providers=[
                ChatgptProvider(
                    model="gpt-3.5-turbo-0613",
                    api_key="sk-123",
                )
            ]
        )
        with pytest.raises(InvalidResponseError):
            flow.completion_with_auto_fixer(
                "What is the captical of france?",
                auto_fixer=my_invalid_auto_fixer,
                auto_fixer_retries=2,
            )
