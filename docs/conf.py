# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

project = 'AEIC'
copyright = '2025, Wyatt Giroux, Prashanth Prakash, Prateek Ranjan, Aditeya Shukla, Raymond Speth'
author = 'Wyatt Giroux, Prashanth Prakash, Prateek Ranjan, Aditeya Shukla, Raymond Speth'
release = '1.0.0a1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary', 
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              'myst_parser']
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

coverage_show_missing_items = True  # Show missing documentation items in the output
coverage_skip_undoc_in_source = False  # Include items without docstrings in the source

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
