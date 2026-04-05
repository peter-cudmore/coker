import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Re-exported symbols have __module__ = 'coker.algebra.dimensions', which
# causes Sphinx to create duplicate object descriptions. Patch them to their
# public module so autodoc uses the canonical coker.* name.
import coker.algebra.dimensions as _dims

for _cls in [_dims.Scalar, _dims.VectorSpace, _dims.FunctionSpace, _dims.Dimension]:
    _cls.__module__ = "coker"

project = "Coker"
author = "Peter Cudmore"
copyright = f"2024, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# Napoleon — Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# autodoc defaults
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# intersphinx — link to numpy/scipy/python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = "Coker"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

