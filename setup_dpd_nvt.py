"""
setup_dpd_nvt.py  –  Build script for the DPD NVT Cython extension.

Usage
-----
    python setup_dpd_nvt.py build_ext --inplace

This produces  dpd_nvt_cy.cpython-3X-x86_64-linux-gnu.so  (or equivalent)
in the current directory, which dpd_nvt_main.py imports automatically.

Requirements
------------
    pip install cython numpy
    apt-get install gcc   # or equivalent C compiler
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="dpd_nvt_cy",
    sources=["dpd_nvt_cy.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3",          # maximum optimisation
        "-march=native",# use all CPU features of the build machine
        "-ffast-math",  # allow reassociation / fused multiply-add etc.
        "-funroll-loops",
    ],
    extra_link_args=[],
    language="c",
)

setup(
    name="dpd_nvt_cy",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
            "nonecheck":      False,
            "initializedcheck": False,
        },
        annotate=True,   # produces dpd_nvt_cy.html showing C-conversion quality
    ),
)
