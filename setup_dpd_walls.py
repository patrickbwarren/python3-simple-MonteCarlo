"""
setup_dpd_walls.py  –  Build script for the DPD walls Cython extension.

Usage
-----
    python setup_dpd_walls.py build_ext --inplace

Produces dpd_walls_cy.cpython-3X-x86_64-linux-gnu.so (or equivalent)
in the current directory.

Requirements
------------
    pip install cython numpy
    gcc (or equivalent C compiler)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="dpd_walls_cy",
    sources=["dpd_walls_cy.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
    ],
    language="c",
)

setup(
    name="dpd_walls_cy",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level":   "3",
            "boundscheck":      False,
            "wraparound":       False,
            "cdivision":        False,  # kept False — floor-based wrap used instead
            "nonecheck":        False,
            "initializedcheck": False,
        },
        annotate=True,
    ),
)
