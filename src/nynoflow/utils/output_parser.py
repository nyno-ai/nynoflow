from typing import Type
from warnings import warn

from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError


def parse_output(data_structure: Type[BaseModel], response: str) -> object:
    """Try to parse the string response to the data structure.

    Raises:
        ValueError: If the response cannot be parsed to the data structure.

    Args:
        data_structure (BaseModel): The data structure to parse the response to.
        response (str): The string response from the command.

    Returns:
        object: The parsed data structure.
    """
    try:
        return data_structure.model_validate_json(response)
    except ValidationError as err:
        warn(f"Failed to parse response to {data_structure.__name__}.")
        raise ValueError from err
