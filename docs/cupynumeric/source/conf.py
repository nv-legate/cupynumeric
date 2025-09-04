# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from os import getenv

import cupynumeric

SWITCHER_PROD = "https://docs.nvidia.com/cupynumeric/switcher.json"
SWITCHER_DEV = "http://localhost:8000/switcher.json"
JSON_URL = SWITCHER_DEV if getenv("SWITCHER_DEV") == "1" else SWITCHER_PROD

ANNOTATE = getenv("LEGATE_ANNOTATION_DOCS") == "1"

# This is the "YY.MM" version string that we want users to see
BASE_VERSION = ".".join(cupynumeric.__version__.split(".", 2)[:2])

# make sure BASE VERSION is formatted as expected
_yy, _mm = BASE_VERSION.split(".")
assert _yy.isdigit()
assert _mm.isdigit()

# -- Project information -----------------------------------------------------

project = "NVIDIA cuPyNumeric"
copyright = "2024, NVIDIA"
author = "NVIDIA Corporation"

if "dev" in cupynumeric.__version__ or "rc" in cupynumeric.__version__:
    # for dev/rc versions just use the entire version with everything, and
    # add it to the page title as well, for easy recognition
    version = release = cupynumeric.__version__
    project += f" ({cupynumeric.__version__})"
else:
    # otherwise, we actually only want the YY.MM to be visible for releases
    version = release = BASE_VERSION

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "nbsphinx",
    "legate._sphinxext.settings",
    "cupynumeric._sphinxext.comparison_table",
    "cupynumeric._sphinxext.implemented_index",
    "cupynumeric._sphinxext.missing_refs",
    "cupynumeric._sphinxext.ufunc_formatter",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------

html_context = {
    # "default_mode": "light",
    "AUTHOR": author,
    "DESCRIPTION": "cuPyNumeric documentation site.",
}

html_static_path = ["_static"]

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "switcher": {
        "json_url": JSON_URL,
        "navbar_start": ["navbar-logo", "version-switcher"],
        "version_match": BASE_VERSION,
    },
    "extra_footer": [
        "The cuPyNumeric project is independent of the CuPy project. CuPy is a trademark of Preferred Networks, Inc, and the name 'cuPyNumeric' is used with their permission.",  # NOQA
        '<script type="text/javascript">if (typeof _satellite !== “undefined”){ _satellite.pageBottom();}</script>',  # NOQA
    ],
    "show_version_warning_banner": True,
}

templates_path = ["_templates"]

# -- Options for extensions --------------------------------------------------

autosummary_generate = True

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

napoleon_custom_sections = [("Availability", "returns_style")]

nbsphinx_execute = "never"

pygments_style = "sphinx"


def setup(app):
    if ANNOTATE:
        app.add_js_file("https://hypothes.is/embed.js", kind="hypothesis")
    app.add_css_file("params.css")
