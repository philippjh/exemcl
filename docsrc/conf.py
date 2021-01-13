extensions = [
    'breathe',
    'exhale',
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "nbsphinx"
]
autodoc_typehints = 'none'
autodoc_docstring_signature = True

breathe_projects = {
    "exemcl": "./doxyoutput/xml"
}
breathe_default_project = "exemcl"
exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ../src"
}

project = "Exemplar-based Clustering"
html_title = "Exemplar-based Clustering Documentation"
html_theme = "sphinx_rtd_theme"
