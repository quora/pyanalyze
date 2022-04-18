Welcome! We'd like to make contributing to pyanalyze as painless
as possible. Here is a quick guide.

It's useful to have a virtual environment to work in. I use
commands like these:
```
$ cd pyanalyze
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ pip install -e .
```

## Black

The code is formatted using [*Black*](https://black.readthedocs.io).
You can run the formatter with:

```
$ black pyanalyze
```

## flake8

We use [flake8](https://flake8.pycqa.org/en/latest/) as a linter:

```
$ flake8 pyanalyze
```

## usort

Imports are sorted using [μsort](https://usort.readthedocs.io/en/stable/index.html):

```
$ usort format pyanalyze
```

## Unit tests

The unit tests are run with [pytest](https://docs.pytest.org/):

```
$ pytest -v pyanalyze
```

Running all of the tests takes a few minutes, so I often use the
`-k` option to select only the tests I am currently working on.
For example:

```
$ pytest -v pyanalyze -k PEP673
```

We run tests on all supported Python versions on GitHub Actions,
but usually I don't bother when testing locally. If necessary, you
can install all supported versions with a tool like
[pyenv](https://github.com/pyenv/pyenv).
