import os
from setuptools import setup
from setuptools.extension import Extension


VERSION = "0.1.0"
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
        long_description="",
        url="https://github.com/quora/pyanalyze",
        license="Apache Software License",
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        keywords="quora static analysis",
        packages=["pyanalyze"],
        install_requires=[
            "attrs",
            "asynq",
            "inspect2",
            "nose",
            "qcore>=0.5.1",
            "ast_decompiler>=0.4.0",
            "six>=1.10.0",
            'futures; python_version < "3.0"',
            'mock; python_version < "3.3"',
            'typeshed_client>=0.4.1; python_version >= "3.6"',
            'enum34; python_version < "3.4"',
            "typing_inspect>=0.5.0",
            'typing; python_version < "3.5"',
            "typing_extensions",
            "mypy_extensions",
            "aenum>=2.2.3",
            "codemod",
        ],
        **setup_kwargs
    )
