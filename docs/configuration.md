# Configuration

The preferred way to configure pyanalyze is using the
`pyproject.toml` configuration file:

```toml
[tool.pyanalyze]
# Paths pyanalyze should check by default
paths = ["my_module/"]
# Paths to import from
import_paths = ["."]

# Enable or disable some checks
possibly_undefined_name = true
duplicate_dict_key = false

# But re-enable it for a specific module
[[tool.pyanalyze.overrides]]
module = "my_module.submodule"
duplicate_dict_key = true
```

It is recommended to always set the following configuration options:

* *paths*: A list of paths (relative to the location of the `pyproject.toml` file) that pyanalyze should check by default.
* *import_paths*: A list of paths (also relative to the configuration file) that pyanalyze should use as roots when trying to import files it is checking. If this is not set, pyanalyze will use entries from `sys.path`, which may produce unexpected results.

Other supported configuration options are listed below.

Almost all configuration options can be overridden for individual modules or packages. To set a module-specific configuration, add an entry to the `tool.pyanalyze.overrides` list (as in the example above), and set the `module` key to the fully qualified name of the module or package.

To see the current value of all configuration options, pass the `--display-options` command-line option:

```
$ python -m pyanalyze --config-file pyproject.toml --display-options
Options:
    add_import (value: True)
    ...
```

Most configuration options can also be set on the command line.

<!-- TODO figure out a way to dynamically include docs for each option -->