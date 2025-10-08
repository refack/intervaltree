from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "intervaltree.interval",
        ["intervaltree/interval.py"],
    ),
    Extension(
        "intervaltree.node",
        ["intervaltree/node.py"],
    ),
    Extension(
        "intervaltree.intervaltree",
        ["intervaltree/intervaltree.py"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    )
)