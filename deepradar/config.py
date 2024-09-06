"""Configuration parsing.

Callers should use `load_config`, which applies the following rules:

- Configuration files (parsed by `load_config`) are provided as a list, which
  is parsed in order; subsequent files override previous ones.
- Each configuration file should be a `.yaml`; these configuration files can
  also use an `!include` tag to include the contents of another `.yaml` or
  `.json` file.
- The include tag can also "merge" its contents into the including mapping
  using `<<: !include file.yaml`.

Example
-------

`a.yaml`::

  a: 5
  b: 10

`b.yaml`::

  b: 15
  x: !include `c.yaml`
  <<: !include `c.yaml`

`c.yaml`::

  c: 20

`load_config("a.yaml", "b.yaml", "c.yaml")`::

  {
    "a": 5,
    "b": 15,
    "c": 20,
    "x": {"c": 20}
  }
"""

import json
import os
import re

import yaml
from beartype.typing import cast


class Loader(yaml.SafeLoader):
    """Yaml loader with override to handle merge keys with include tags.

    This class adds the `!include` tag, which can be used as an ordinary tag,
    or in a merge tag.
    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]  # type: ignore
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(
            self._root, self.construct_scalar(node))  # type: ignore

        with open(filename, 'r') as f:
            if filename.endswith('.yaml'):
                return yaml.load(f, Loader)
            elif filename.endswith('.json'):
                return json.load(f)
            else:
                return f.read()

    def expand(self, node):
        if isinstance(node, yaml.ScalarNode):
            return [self.construct_scalar(node)]
        elif isinstance(node, yaml.SequenceNode):
            return self.construct_sequence(node)
        elif isinstance(node,  yaml.MappingNode):
            contents = self.construct_mapping(node, deep=True)

            def _walk(obj, prefix: list[str] = []) -> list[str]:
                if isinstance(obj, list):
                    return [os.path.join(*prefix, x) for x in obj]
                elif isinstance(obj, dict):
                    res = []
                    for k, v in obj.items():
                        res += _walk(v, prefix=prefix + [k])
                    return res
                else:
                    return os.path.join(*prefix, obj)  # type: ignore

            return _walk(contents)
        else:
            raise ValueError("!expand must be a sequence, mapping, or scalar.")

    def construct_mapping(self, node, deep=False):
        _mapping = {}
        for key, value in node.value:
            is_merge_include = (
                key.tag == 'tag:yaml.org,2002:merge'
                and value.tag == '!include')
            if is_merge_include:
                _mapping.update(cast(dict, self.include(value)))
            else:
               _key = self.construct_object(key, deep=deep)
               _value = self.construct_object(value, deep=deep)
               _mapping[_key] = _value

        return _mapping


Loader.add_constructor('!include', Loader.include)
Loader.add_constructor('!expand', Loader.expand)


def merge_config(x: dict, y: dict) -> None:
    """Merge two configurations in-place.

    Args:
        x: target configuration dict.
        y: config to be merged in.
    """
    for k, v in y.items():
        if k in x:
            if isinstance(v, dict) and isinstance(x[k], dict):
                merge_config(x[k], v)
            elif isinstance(v, list) and isinstance(x[k], list):
                x[k] = x[k] + v
            else:
                x[k] = v
        else:
            x[k] = v


def parse_paths(configs: list[str]) -> list[str]:
    """Expand shorthand for configuration files."""

    def _add_ext(s: str) -> str:
        if not s.endswith('.json') and not s.endswith('.yaml'):
            return s + ".yaml"
        else:
            return s

    res = []
    for c in configs:
        expandable = re.match(r"^(.*)\[(.*)\]$", c)
        if expandable:
            base = expandable.group(1)

            default = os.path.join(base, os.path.basename(base))
            if os.path.exists(default + ".yaml"):
                res.append(default)
            for modifier in expandable.group(2).split(','):
                res.append(os.path.join(base, modifier))
        else:
            res.append(c)
    return [_add_ext(s) for s in res]


def load_config(configs: list[str]) -> dict:
    """Load a list of configuration files."""
    res: dict = {}
    for path in parse_paths(configs):
        with open(path) as f:
            cfg = yaml.load(f, Loader)
        merge_config(res, cfg)
    return res
