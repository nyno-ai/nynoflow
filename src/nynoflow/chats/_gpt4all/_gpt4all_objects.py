from typing import TypedDict


class GPT4AllModel(TypedDict):
    """This is a model object from the list_models response of gpt4all api.

    The mixed types ignored are because we have no control over
    the response of gpt4all api and this is the response object.
    """

    order: str
    md5sum: str
    name: str
    filename: str
    filesize: str
    requires: str
    ramrequired: str
    parameters: str
    quant: str
    type: str
    systemPrompt: str  # noqa
    description: str
    url: str
    promptTemplate: str  # noqa
