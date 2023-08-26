from pytest_mock import MockerFixture

from nynoflow.chats import Chat
from nynoflow.chats._gpt4all import Gpt4AllProvider
from nynoflow.chats.chat_types import ChatRequest
from nynoflow.util import logger


def test_chatgpt(mocker: MockerFixture) -> None:
    """This is a test for the chatgpt function.

    Args:
        mocker: The mocker object.
    """
    logger.debug("Testing GPT4All")
    chat = Chat(
        providers=[
            Gpt4AllProvider(
                model_name="orca-mini-3b.ggmlv3.q4_0.bin", allow_download=True
            )
        ]
    )
    res = chat.completion(
        ChatRequest(
            role="user",
            content="What is the captial of france?",
        )
    )
    logger.debug(res)
    assert res
