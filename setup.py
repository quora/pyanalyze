import os
from pathlib import Path
from setuptools import setup
from setuptools.extension import Extension


VERSION = "0.5.0"
# Used in internal packaging system.
if "SANTA_PACKAGE_VERSION" in os.environ:
    VERSION = "%s.%s" % (VERSION, os.environ["SANTA_PACKAGE_VERSION"])

    CYTHON_MODULES = ["name_check_visitor"]
    DATA_FILES = ["%s.pxd" % module for module in CYTHON_MODULES]
    EXTENSIONS = [
        Extension("pyanalyze.%s" % module, ["pyanalyze/%s.py" % module])
        for module in CYTHON_MODULES
    ]
    setup_kwargs = {
        "package_data": {"pyanalyze": DATA_FILES},
        "ext_modules": EXTENSIONS,
        "setup_requires": "Cython",
    }
else:
    setup_kwargs = {}


if __name__ == "__main__":
    setup(
        name="pyanalyze",
        version=VERSION,
        author="Quora, Inc.",
        author_email="jelle@quora.com",
        description="A static analyzer for Python",
        entry_points={"console_scripts": ["pyanalyze=pyanalyze.__main__:main"]},
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/quora/pyanalyze",
        license="Apache Software License",
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        keywords="quora static analysis",
        packages=["pyanalyze"],
        install_requires=[
            "attrs",
            "asynq",
            "dataclasses; python_version < '3.7'",
            "qcore>=0.5.1",
            "ast_decompiler>=0.4.0",
            "typeshed_client>=1.0.0,<2",
            "typing_inspect>=0.7.0",
            "typing_extensions",
            "mypy_extensions",
            "aenum>=2.2.3",
            "codemod",
        ],
        **setup_kwargs,
    )
