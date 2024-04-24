# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Configuration file for the Sphinx documentation builder."""

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "ArePyTools"
copyright = "2024, Aresys S.R.L."
author = "Aresys S.R.L."

import arepytools

arepytools_version = arepytools.__version__
# The short X.Y version
version = arepytools_version[
    : arepytools_version[: arepytools_version.rfind(".")].rfind(".")
]
# The full version, including alpha/beta/rc tags
release = arepytools_version

# -- General configuration ---------------------------------------------------

needs_sphinx = "7.2"
extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",  # for numpy docstring
    "sphinx.ext.mathjax",
    "nbsphinx",
]
python_use_unqualified_type_names = True
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/aresys-srl/arepytools",
    "icon_links": [
        {
            "name": "Aresys",
            "url": "https://www.aresys.it/",
            "icon": "_static/icons/aresys_logo.svg",
            "type": "local",
        }
    ],
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_title = project
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_copy_source = False

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "ArePyToolsdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_documents = [
    (
        master_doc,
        "ArePyTools.tex",
        "ArePyTools Documentation",
        "Aresys S.R.L.",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "arepytools", "ArePyTools Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "ArePyTools",
        "ArePyTools Documentation",
        author,
        "ArePyTools",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

autodoc_default_options = {"members": True, "undoc-members": True}
autodoc_member_order = "bysource"  # alphabetical, groupwise
autoclass_content = "both"  # class, init, both
autodoc_preserve_defaults = True

autodoc_type_aliases = {
    "npt.ArrayLike": "ArrayLike",
    "ReferenceFrameLike": "ReferenceFrameLike",
    "RotationOrderLike": "RotationOrderLike",
}

napoleon_use_param = True
napoleon_preprocess_types = True
napoleon_type_aliases = {}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


from arepytools.timing.precisedatetime import PreciseDateTime

rst_prolog = """
.. |PRECISEDATETIME_REFERENCE_TIME| replace:: ``{PRECISEDATE_TIME_REFERENCE_TIME}``
.. |PRECISEDATETIME_1985| replace:: ``{PRECISEDATETIME_1985}``
""".format(
    PRECISEDATE_TIME_REFERENCE_TIME=PreciseDateTime.get_reference_datetime(),
    PRECISEDATETIME_1985=str(PreciseDateTime.from_sec85(0)).split(" ")[0],
)
