# NynoFlow

[![PyPI](https://img.shields.io/pypi/v/nynoflow.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/nynoflow.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/nynoflow)][python version]
[![License](https://img.shields.io/pypi/l/nynoflow)][license]

[![Read the documentation at https://nynoflow.readthedocs.io/](https://img.shields.io/readthedocs/nynoflow/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/nyno-ai/nynoflow/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/nyno-ai/nynoflow/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/nynoflow/
[status]: https://pypi.org/project/nynoflow/
[python version]: https://pypi.org/project/nynoflow
[read the docs]: https://nynoflow.readthedocs.io/
[tests]: https://github.com/nyno-ai/nynoflow/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/nyno-ai/nynoflow
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

Nynoflow was built out of frustration with current tooling to build LLM applications. Some focus too much on UI building, which is limited at best. Some over-abstract on certain areas to the point that you can't understand what's the prompt being sent to the LLM. So here at nyno-ai, we set to build the best framework and tooling to build LLM applications for developers.

## Features

- TODO

## Requirements

- TODO

## Installation

You can install _NynoFlow_ via [pip] from [PyPI]:

```console
$ pip install nynoflow
```

## Usage

- TODO

## Defaults

Here are all the default configuations you should make sure to review when creating an app with nynoflow:

| Config                       | Description                                                                                                                                                                                                                                                                                             | Default |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| flow.completion.token_offset | This is used for the historical message cutoff (removing old messages when the token limit is exceeded). This number is the number of tokens you may require for the answer. This is deliberetly different then the token_limit because you may get more token space since messages are cut as a whole. | 16      |
|                              |                                                                                                                                                                                                                                                                                                         |         |
|                              |                                                                                                                                                                                                                                                                                                         |         |

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_NynoFlow_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

<!-- github-only -->

[license]: https://github.com/nyno-ai/nynoflow/blob/main/LICENSE
[contributor guide]: https://github.com/nyno-ai/nynoflow/blob/main/CONTRIBUTING.md
[command-line reference]: https://nynoflow.readthedocs.io/en/latest/usage.html
