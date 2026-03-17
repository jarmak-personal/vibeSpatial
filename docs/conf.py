"""Sphinx configuration for vibeSpatial documentation."""

project = "vibeSpatial"
copyright = "2025, vibeSpatial Contributors"
author = "vibeSpatial Contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "vibeproj": ("https://jarmak-personal.github.io/vibeProj/", None),
    "vibespatial-raster": ("https://jarmak-personal.github.io/vibespatial-raster/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "html_admonition",
    "attrs_inline",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "ops", "decisions/index.md"]

# -- Theme: Furo + NEON GRID overlay -----------------------------------------
html_theme = "furo"
html_title = "vibeSpatial"

html_static_path = ["_static"]
html_css_files = ["css/vibespatial.css"]
html_js_files = ["js/vibespatial.js"]

html_theme_options = {
    "source_repository": "https://github.com/vibeSpatial/vibeSpatial",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {},
    "dark_css_variables": {},
}

# Force dark mode default
html_context = {
    "default_mode": "dark",
}
