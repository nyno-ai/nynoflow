import inspect
from typing import Any, Callable, Generic, Literal, TypeVar, get_type_hints
from warnings import warn

import jsonschema  # type: ignore
from attrs import define, field
from docstring_parser import parse
from pydantic import create_model

from nynoflow.exceptions import (
    InvalidFunctionCallResponseError,
    MissingDescriptionError,
    MissingDocstringError,
    MissingTypeHintsError,
)


FunctionReturnType = TypeVar("FunctionReturnType")


@define
class Function(Generic[FunctionReturnType]):
    """Implements the functions for the LLM apps.

    Args:
        name (str): The name of the function.
        description (str): The description of the function.
        func (Callable[..., FunctionReturnType]): The function to invoke.
        parameters (dict[str, Any]): The parameters of the function.
        retries_on_error (int, optional): Number of times to reprompt if the LLM responds with bad invocation data. Defaults to 0.
    """

    name: str = field()
    description: str = field()
    func: Callable[..., FunctionReturnType] = field()
    parameters: dict[str, Any] = field()
    retries_on_error: int = field(default=0)

    def invoke(self, **kwargs: Any) -> FunctionReturnType:
        """Invoke the function with the given parameters.

        Raises:
            InvalidFunctionCallResponseError: If the function call from the LLM is invalid.

        Args:
            kwargs: The parameters to pass to the function.

        Returns:
            FunctionReturnType: The return value of the function.
        """
        # Verify the arguments and their types againts the parameters schema
        try:
            jsonschema.validate(kwargs, self.parameters)
        except jsonschema.exceptions.ValidationError as err:
            raise InvalidFunctionCallResponseError(
                f"Function call from LLM is invalid for function {self.name}."
            ) from err

        return self.func(**kwargs)

    @classmethod
    def from_function(
        cls, func: Callable[..., FunctionReturnType]
    ) -> "Function[FunctionReturnType]":
        """Parse a function docstring and type hints to get the name, description and parameters.

        The parsing is done in the following way:
        - Name: The name of the function.
        - Description: The short description and the long description of the function together.
        - Parameters:
            - Name: The argument name.
            - Type: The type hint of the argument.
            - Description: The description of the argument from the docstring.
        - Required: Parameters that do not have a default value.

        In case you want to provide a more detailed type, please initialize the class manually with
        the json schema for the parameters to utilize the full json schema specification.

        Raises:
            MissingDocstringError: If the function does not have a docstring.
            MissingDescriptionError: If the function does not have a description.
            MissingTypeHintsError: If the function does not have type hints for some arguments.

        Args:
            func (FuncType): The function to parse.

        Returns:
            Function: The Function instance with the parsed docstring.
        """
        function_name: str = func.__name__

        docstring = func.__doc__
        if docstring is None:
            raise MissingDocstringError()

        parsed_docstring = parse(docstring)

        description = " ".join(
            [
                parsed_docstring.short_description or "",
                parsed_docstring.long_description or "",
            ]
        ).strip()

        if description == "":
            raise MissingDescriptionError(
                f"No description found for function {function_name} when parsing docstring."
            )

        # Get all the function arguments
        all_params = inspect.signature(func).parameters

        # Get all the type hints for the function arguments that have type hints
        type_hints = get_type_hints(func)

        missing_hints = set(all_params.keys()) - set(type_hints.keys()) - {"return"}
        # Verify that all parameters have a type hint
        if missing_hints:
            raise MissingTypeHintsError(
                f"Missing type hints for parameters: {', '.join(missing_hints)}"
            )

        # Extract parameter descriptions from docstring
        param_descriptions = {
            param.arg_name: param.description for param in parsed_docstring.params
        }

        fields: dict[str, Any] = {}
        schema_extra: dict[
            Literal["properties"], dict[str, dict[Literal["description"], str]]
        ] = {"properties": {}}
        for name, param in all_params.items():
            # Get type hint for the parameter
            param_type = type_hints.get(name)

            # Get default value for the parameter
            default_value = param.default if param.default is not param.empty else ...

            fields[name] = (param_type, default_value)

            # Get parameter description from docstring
            param_description = param_descriptions.get(name)
            if param_description is None or param_description == "":
                warn(
                    f"No description found for parameter {name} when parsing docstring."
                    "This may cause the LLM to provide incorrect parameters."
                )
                param_description = str(name)

            schema_extra["properties"][name] = {"description": param_description}

        # Create Pydantic model using create_model
        parameters_model = create_model("parameters", **fields)
        parameters_model.schema_extra = schema_extra
        parameters = parameters_model.model_json_schema()

        return cls(
            name=function_name,
            description=description,
            func=func,
            parameters=parameters,
        )
