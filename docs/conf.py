"""Sphinx configuration for vibeSpatial documentation."""

project = "vibeSpatial"
copyright = "2026, vibeSpatial Contributors"
author = "vibeSpatial Contributors"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

# -- sphinx-autoapi: static-analysis API reference ----------------------------
autoapi_dirs = ["../src/vibespatial"]
autoapi_type = "python"
autoapi_root = "autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = [
    "*/cuda/*",
    "*/bench/*",
    "*/testing/*",
    # Internal overlay submodules fully re-exported through __init__.py.
    "*/overlay/gpu.py",
    "*/overlay/reconstruction.py",
    "*/overlay/dissolve.py",
]
autoapi_keep_files = True
# Let our curated docs/user/api.md serve as the entry point; autoapi adds
# its own toctree entry to the root for the full generated reference.
autoapi_add_toctree_entry = True

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

# Suppress cosmetic warnings:
# - duplicate: re-exports create duplicate py:function entries
# - ref.python: ambiguous cross-refs from re-exported symbols
# - ref.ref/ref.doc: GeoPandas docstrings reference upstream labels/docs
# - image.not_readable: GeoPandas docstrings reference upstream SVGs
suppress_warnings = [
    "duplicate",
    "ref.python",
    "ref.ref",
    "ref.doc",
    "image.not_readable",
    "toc.not_included",
    "docutils",
    "autoapi.python_import_resolution",
]

# -- Theme: Furo + NEON GRID overlay -----------------------------------------
html_theme = "furo"
html_title = "vibeSpatial"

html_static_path = ["_static"]
html_css_files = ["css/vibespatial.css"]
html_js_files = ["js/vibespatial.js"]

html_theme_options = {
    "source_repository": "https://github.com/jarmak-personal/vibeSpatial",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jarmak-personal/vibeSpatial",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/vibespatial/",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 17 20"><path d="M8.5 0L0 4.7v9.5L8.5 19l8.5-4.8V4.7L8.5 0zm0 2.2l5.9 3.3-5.9 3.3-5.9-3.3 5.9-3.3zM1.5 6l6.2 3.5v7L1.5 13V6z"></path></svg>',
            "class": "",
        },
    ],
    "light_css_variables": {},
    "dark_css_variables": {},
}

# Force dark mode default
html_context = {
    "default_mode": "dark",
}
