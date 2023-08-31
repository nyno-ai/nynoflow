from warnings import warn

import tiktoken
from attrs import define, field

from nynoflow.chats._chatgpt._chatgpt_objects import (  # ChatgptRequestFunction,
    ChatgptMessageHistory,
)


@define
class ChatgptTiktokenTokenizer:
    """Tokenizer for ChatGPT using TikToken.

    Args:
        model (str): The model to use.
    """

    model: str = field()

    _encoding: tiktoken.Encoding = field(init=False)
    _tokens_per_message: int = field(init=False)
    _tokens_per_name: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Configure the TikToken tokenizer.

        Taken from here: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            warn(f"Model {self.model} not found. Using cl100k_base encoding.")
            self._encoding = tiktoken.get_encoding("cl100k_base")

        if self.model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            self._tokens_per_message = 4
            self._tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            self._tokens_per_message = 3
            self._tokens_per_name = 1

    # def _calculate_function_tokens(
    #     self, functions: list[ChatgptRequestFunction]
    # ) -> int:
    #     """Return the number of tokens used by a list of functions.

    #     Taken from here: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11

    #     Args:
    #         functions (ChatgptRequestFunctions, optional): The functions to count the tokens of. Defaults to None.

    #     Returns:
    #         int: The number of tokens used by the messages.
    #     """
    #     token_count = 0
    #     for function in functions:
    #         token_count += len(self._encoding.encode(function["name"]))
    #         token_count += len(self._encoding.encode(function["description"]))
    #         properties = function.get("parameters", {}).get("properties", {})
    #         logger.debug(f"Properties: {properties} {len(properties)}")

    #         if len(properties) > 0:
    #             token_count += 11

    #         for property_key in properties:
    #             token_count += len(self._encoding.encode(property_key))
    #             v = properties[property_key]
    #             for property_field in v:
    #                 if property_field == "type":
    #                     token_count += 2
    #                     token_count += len(self._encoding.encode(v["type"]))
    #                 elif property_field == "description":
    #                     token_count += 2
    #                     token_count += len(self._encoding.encode(v["description"]))
    #                 elif property_field == "enum":
    #                     token_count -= 3
    #                     for o in v["enum"]:
    #                         token_count += 3
    #                         token_count += len(self._encoding.encode(o))
    #                 else:
    #                     print(f"Warning: not supported property_field {property_field}")

    #     token_count += 12
    #     return token_count

    def _calculate_messages_tokens(self, messages: ChatgptMessageHistory) -> int:
        """Return the number of tokens used by a list of messages.

        Args:
            messages (ChatgptMessageHistory): The messages to count the tokens of.

        Returns:
            int: The number of tokens used by the messages.
        """
        token_count = 0
        for message in messages:
            message_tokens = self._tokens_per_message

            for key, value in message.items():
                if value is not None:
                    message_tokens += len(self._encoding.encode(str(value)))
                if key == "name":
                    message_tokens += self._tokens_per_name
            token_count += message_tokens

        token_count += 3  # every reply is primed with <|start|>assistant<|message|>
        return token_count

    def token_count(
        self,
        messages: ChatgptMessageHistory,
        # functions: list[ChatgptRequestFunction] | None = None,
    ) -> int:
        """Return the number of tokens used by a list of messages and functions.

        Args:
            messages (ChatgptMessageHistory): The messages to count the tokens of.

        Returns:
            int: The number of tokens used by the messages.
        """
        token_count = self._calculate_messages_tokens(messages)
        # if functions:
        #     logger.debug("We have some functions yo")
        #     token_count += 9
        #     token_count += self._calculate_function_tokens(functions)

        # else:
        #     logger.debug("No functions why")

        return token_count
