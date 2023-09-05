from typing import Type

from pydantic import BaseModel

from nynoflow.exceptions import InvalidOutputError


def output_parser(data_structure: Type[BaseModel], response: str) -> object:
    """Try to parse the string response to the data structure.

    Raises:
        InvalidOutputError: The output parser recieved invalid output.

    Args:
        data_structure (BaseModel): The data structure to parse the response to.
        response (str): The string response from the command.

    Returns:
        object: The parsed data structure.
    """
    try:
        return data_structure.model_validate_json(response)
    except ValueError as err:
        raise InvalidOutputError() from err
