# -*- coding: utf-8 -*-
#

import sys
import os
import shlex

# General information about the project.
project = u'Epsilon'
copyright = u'2016, Matt Wytock'
author = u'Matt Wytock'
version = u'0.2.4'
release = u'0.2.4'

# Settings
extensions = [
    'sphinx.ext.mathjax',
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = None
exclude_patterns = ['_build']
todo_include_todos = False

# HTML output
html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'description': 'Scalable convex programming with fast linear and proximal operators.',
    'github_user': 'mwytock',
    'github_repo': 'epsilon',
    'github_button': True,
    'analytics_id': 'UA-72208233-1',
    'code_font_size': '0.8em'
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
    ]
}
html_show_copyright = False
