Orange3 Interactions Add-on
===========================

This is be an Interactions add-on for [Orange3](http://orange.biolab.si). For now it extends Orange only in the scripting part.

Installation
------------

To install the add-on, run

    pip install .

from the root directory.

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    pip install -e .

Documentation / widget help can be built by running

    make html htmlhelp

from the doc directory. You need [sphinx](http://www.sphinx-doc.org/en/stable/index.html) to generate the docs. Once generated you can
find them in the build directory.

Usage
-----

For usage look at the generated html doc pages. They contain a short turorial.

This work is part of a summer internship at the laboratory for Bioinformatics at FRI Ljubljana.
