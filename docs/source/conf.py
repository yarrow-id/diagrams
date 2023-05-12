# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'yarrow'
copyright = '2023, Paul Wilson'
author = 'Paul Wilson'

# Ensure the yarrow module is in the path so autosummary can load it.
import sys
from pathlib import Path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme", # readthedocs theme
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
]

# class members should be in the order they are written in the file
autodoc_member_order = "bysource"

# references
bibtex_bibfiles = ["refs.bib"]

# let autosummary recurse and generate all modules specified
# https://stackoverflow.com/questions/2701998/
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
