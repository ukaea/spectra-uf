# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spectrauf'
copyright = '2025, R. Worrall'
author = 'R. Worrall'
release = '0.1'

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
    "sphinx_copybutton",    # for inline code copy button
]

# Tell MyST to parse dollar math
myst_enable_extensions = [
    "dollarmath",   # allows $...$ and $$...$$
    "amsmath",      # allows LaTeX math environments
]

autosummary_generate = False  # make Sphinx create stub pages automatically

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

html_sidebars = {
    "**": ["search-field.html", "globaltoc.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
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

import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # so autodoc can find your package
