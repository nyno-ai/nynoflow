import openai
from gpt4all import GPT4All

from nynoflow.util import logger
from nynoflow.utils.cl100k_base_token_counter import (
    cl100k_base_num_tokens_from_messages,
)
from nynoflow.utils.openai_token_counter import openai_num_tokens_from_messages
from tests.conftest import ConfigTests


def test_token_counter_openai(config: ConfigTests) -> None:
    """Test the token counter."""
    example_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello there",
        },
    ]

    for model in [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4",
    ]:
        logger.debug(model)
        # example token count from the function defined above
        logger.debug(
            f"""{openai_num_tokens_from_messages(example_messages, model)} prompt
            tokens counted by openai_num_tokens_from_messages()."""
        )
        # example token count from the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=example_messages,
            temperature=0,
            max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
        )
        logger.debug(
            f'{response["usage"]["prompt_tokens"]} prompt tokens counted by the OpenAI API.'
        )

        assert response["usage"]["prompt_tokens"] == openai_num_tokens_from_messages(
            example_messages, model
        )


def test_token_counter_gpt4all(config: ConfigTests) -> None:
    """Test the token counter for gpt4all."""
    prompt = "Hello"
    gpt4all_client = GPT4All(
        model_name="orca-mini-3b.ggmlv3.q4_0.bin",
        allow_download=True,
    )

    official_token_count = 0
    response = ""
    for _ in range(1):
        for token in gpt4all_client.generate(prompt, streaming=True):
            official_token_count += 1
            response += token
        calculated_token_count = cl100k_base_num_tokens_from_messages(
            [response], gpt4all_client.config["path"]
        )
        logger.debug(f"Token count: {calculated_token_count}")

        assert official_token_count == calculated_token_count
