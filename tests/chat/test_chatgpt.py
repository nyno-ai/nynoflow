from pytest_mock import MockerFixture

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt import ChatgptProvider
from nynoflow.chats._chatgpt import ChatgptRequest
from nynoflow.chats._chatgpt import ChatgptRequestMessage
from nynoflow.chats._chatgpt import ChatgptResponse
from nynoflow.chats._chatgpt import ChatgptResponseChoice
from nynoflow.chats._chatgpt import ChatgptResponseMessage
from nynoflow.chats._chatgpt import ChatgptResponseUsage
from nynoflow.chats.chat_types import ChatRequest
from nynoflow.util import logger


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


def test_chatgpt(mocker: MockerFixture) -> None:
    """This is a test for the chatgpt function.

    Args:
        mocker: The mocker object.
    """
    logger.debug("Testing Chatgpt")

    mocker.patch("openai.ChatCompletion.create", return_value=chatgpt_response)
    chat = Chat(
        providers=[
            ChatgptProvider(
                organization="myorg",
                api_key="sk-123",
                model="gpt-3.5-turbo",
            )
        ]
    )
    res = chat.completion(
        ChatRequest(role="user", content="What is the captial of france?")
    )
    assert res