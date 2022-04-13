import os
from pathlib import Path
from setuptools import setup
from setuptools.extension import Extension


version = "0.7.0"
package_data = ["test.toml", "stubs/*/*.pyi"]
# Used in internal packaging system.
if "SANTA_PACKAGE_VERSION" in os.environ:
    CYTHON_MODULES = ["name_check_visitor"]
    DATA_FILES = [f"{module}.pxd" for module in CYTHON_MODULES]
    EXTENSIONS = [
        Extension(f"pyanalyze.{module}", [f"pyanalyze/{module}.py"])
        for module in CYTHON_MODULES
    ]
    package_data += DATA_FILES
    setup_kwargs = {"ext_modules": EXTENSIONS, "setup_requires": "Cython"}
else:
    setup_kwargs = {}


if __name__ == "__main__":
    setup(
        name="pyanalyze",
        version=version,
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
            "asynq",
            "dataclasses; python_version < '3.7'",
            "qcore>=0.5.1",
            "ast_decompiler>=0.4.0",
            "typeshed_client>=2.0.0",
            "typing_inspect>=0.7.0",
            "typing_extensions>=4.0.0",
            "aenum>=2.2.3",
            "codemod",
            "tomli>=1.1.0",
        ],
        # These are useful for unit tests of pyanalyze extensions
        # outside the package.
        package_data={"pyanalyze": package_data},
        **setup_kwargs,
    )
