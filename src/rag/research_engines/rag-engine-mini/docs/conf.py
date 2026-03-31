# Sphinx Documentation Configuration
# For RAG Engine Mini

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath("../.."))

# Project Information
project = "RAG Engine Mini"
copyright = f"{datetime.now().year}, RAG Engine Team"
author = "RAG Engine Team"
version = "1.0.0"
release = "1.0.0"

# General Configuration
extensions = [
    "sphinx.ext.autodoc",  # Auto-document from docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.todo",  # Support for todo items
    "sphinx.ext.coverage",  # Documentation coverage checking
    "sphinx.ext.githubpages",  # GitHub Pages support
    "myst_parser",  # Markdown support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source Suffix
source_suffix = {
    ".rst": None,
    ".md": None,
}

# Master Document
master_doc = "index"

# Language
language = "en"

# HTML Output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# HTML Theme Options
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Autodoc Configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon Configuration (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx Configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

# Todo Configuration
todo_include_todos = True

# Coverage Configuration
coverage_show_missing_items = True

# HTML Context
html_context = {
    "display_github": True,
    "github_user": "your-org",
    "github_repo": "rag-engine-mini",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add any paths that contain custom static files
html_static_path = ["_static"]


# Custom CSS
def setup(app):
    app.add_css_file("custom.css")
