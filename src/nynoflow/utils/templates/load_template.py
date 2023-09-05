import os

from jinja2 import Template

from nynoflow.chats.function import Function


def read_template(template_name: str) -> Template:
    """Read a template file.

    Args:
        template_name (str): The name of the template file to read.

    Returns:
        Template: The contents of the template file.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, template_name)
    with open(file_path) as f:
        template = Template(f.read())

    return template


def render_output_formatter(prompt: str, json_schema_format: str) -> str:
    """Render the output formatter template.

    Args:
        prompt (str): The prompt to be used in the output formatter.
        json_schema_format (str): The format of the JSON schema to be used in the output formatter.

    Returns:
        str: The rendered output formatter template.
    """
    template = read_template("output_formatter.j2")
    return template.render(prompt=prompt, json_schema_format=json_schema_format)


def render_functions(prompt: str, functions: list[Function]) -> str:
    """Render the functions template.

    Args:
        prompt (str): The prompt to be used in the functions template.
        functions (list[Function]): The functions to be used in the functions template.

    Returns:
        str: The rendered functions template.
    """
    template = read_template("functions.j2")
    return template.render(prompt=prompt, functions=functions)


def render_function_call(prompt: str, function: Function) -> str:
    """Render the function call template.

    Args:
        prompt (str): The prompt to be used in the function call template.
        function (Function): The function to be used in the function call template.

    Returns:
        str: The rendered function call template.
    """
    template = read_template("function_call.j2")
    return template.render(prompt=prompt, function=function)
