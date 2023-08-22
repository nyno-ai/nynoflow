"""Sphinx configuration."""
project = "NynoFlow"
author = "nyno.ai"
copyright = "2023, nyno.ai"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
