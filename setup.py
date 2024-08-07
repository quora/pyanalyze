from pathlib import Path

from setuptools import setup

version = "0.13.1"
package_data = ["test.toml", "stubs/*/*.pyi"]


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
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        keywords="quora static analysis",
        packages=["pyanalyze"],
        install_requires=[
            "asynq",
            "qcore>=0.5.1",
            "ast_decompiler>=0.4.0",
            "typeshed_client>=2.1.0",
            "typing_extensions>=4.12.0",
            "codemod",
            "tomli>=1.1.0",
        ],
        extras_require={
            "tests": [
                "pytest",
                "mypy_extensions",
                "attrs",
                "pydantic",
                "annotated-types",
            ]
        },
        # These are useful for unit tests of pyanalyze extensions
        # outside the package.
        package_data={"pyanalyze": package_data},
        python_requires=">=3.8",
    )
