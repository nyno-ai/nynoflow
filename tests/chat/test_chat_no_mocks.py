from pydantic import BaseModel

from nynoflow.chats import Chat
from nynoflow.chats._chatgpt._chatgpt import ChatgptProvider
from tests.conftest import ConfigTests


class TestChatNoMocks:
    """Test the Chat class without mocks."""

    def test_output_formatter(self, config: ConfigTests) -> None:
        """Test the output formatter to make sure the output is as expected."""
        chat = Chat(
            providers=[
                ChatgptProvider(
                    api_key=config["OPENAI_API_KEY"],
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                )
            ]
        )

        class Person(BaseModel):
            """This is a person."""

            first_name: str
            last_name: str

        output = chat.completion("My friend name is john lennon.", output_format=Person)
        assert output
