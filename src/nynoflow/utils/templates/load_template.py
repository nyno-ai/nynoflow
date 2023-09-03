import os

from jinja2 import Template


def render_output_formatter(prompt: str, json_schema_format: str) -> str:
    """Render the output formatter template.

    Args:
        prompt (str): The prompt to be used in the output formatter.
        json_schema_format (str): The format of the JSON schema to be used in the output formatter.

    Returns:
        str: The rendered output formatter template.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, "output_formatter.j2")
    with open(file_path) as f:
        template = Template(f.read())

    return template.render(prompt=prompt, json_schema_format=json_schema_format)
