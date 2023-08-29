# import tiktoken
import sentencepiece
from transformers import AutoTokenizer

from nynoflow.util import logger


def cl100k_base_num_tokens_from_messages(messages: list[str], model_file) -> int:
    """Return the number of tokens used by a list of messages. Uses the encoding encoding cl100k_base.

    Args:
        messages (list[dict[str, str]]): The messages to count the tokens of.
        model_file (str): Model file path

    Returns:
        int: The number of tokens used by the messages.
    """
    # encoding = tiktoken.get_encoding("cl100k_base")
    # num_tokens = sum([len(encoding.encode(message)) for message in messages])
    # return num_tokens

    sp = sentencepiece.SentencePieceProcessor(model_file=model_file)
    logger.info(sp)
    prompt_tokens = sp.encode_as_ids("".join(messages))
    return len(prompt_tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    logger.debug(tokenizer)
    result = tokenizer("".join(messages), return_tensors="pt")
    logger.debug(result)
    return len(result)
