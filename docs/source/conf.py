import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import PackageNotFoundError, version as _pkg_version

project = 'spectrauf'
copyright = '2025, UKAEA'      # was 'R. Worrall'; pyproject.toml says UKAEA, so pick one
author = 'UKAEA'

try:
    release = _pkg_version("spectrauf")                # full string, e.g. "0.1.0b1"
except PackageNotFoundError:
    release = "unknown"
version = ".".join(release.split(".")[:2])             # short form, e.g. "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_static_path = ["_static"]

def setup(app):
    app.add_css_file("custom.css")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "myst_parser",          # if you want Markdown support
    "sphinx.ext.mathjax",   # enable MathJax rendering
    "sphinx_copybutton",    # for inline code copy button,
    "sphinx_autodoc_typehints"
]

# Tell MyST to parse dollar math
myst_enable_extensions = [
    "dollarmath",   # allows $...$ and $$...$$
    "amsmath",      # allows LaTeX math environments
]

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'examples/*.py']

autodoc_default_options = {
    'no-index': True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

# Enable automatic numbering for figures, tables, and code-blocks
numfig = True

# Optional: Customize the format
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
}

# Math settings (usually part of sphinx.ext.mathjax)
math_number_all = True  # Set to True to number EVERY equation


sys.path.insert(0, os.path.abspath(".."))  # so autodoc can find your package
