from typing import Union

from .base_tokenizer import BaseTokenizer
from .openai_tokenizer import OpenAITokenizer


Tokernizers = Union[BaseTokenizer, OpenAITokenizer]

__all__ = [
    "BaseTokenizer",
    "OpenAITokenizer",
]
