"""Configuration parsing.

Callers should use `load_config`, which has additional features compared to
a generic `yaml.Load` to allow for more expressive configuration systems.


Configuration Merging
---------------------

Configuration files (parsed by `load_config`) are provided as a list, which
is parsed in order; subsequent ("right") files override previous ("left") ones.
Overriding is done by expanding the tree of the left and right configurations.
Then, at each level:

- If the left and right nodes are both mappings, they are recursively merged.
- If the left and right nodes are both sequences, the right node is appended to
  the left node.
- Otherwise, the right node overwrites the left node. This includes cases
  where both nodes are scalars, but also allows scalars to overwrite mappings
  and sequences.
- Exception: if the right node's name starts is given the prefix `:`, it
  overwrites the left node completely in all cases

For example::

  left.yaml:
    a: 5
    b:
      x: 10
    c:
    - 15
    d:
    - 0

  right.yaml:
    a2: 6
    b:
      x: 11
    c:
    - 16
    :d: 5

  load_config(["left.yaml", "right.yaml"]):
    {"a": 5, "a2": 6, "b": {"x": 11}, "c": [15, 16], "d": 5}


File Shorthand
--------------

Instead of needing to specify the full path of each configuration file, a
shorthand can be used, which is automatically expanded by `load_config`.

- `file/path -> file/path.yaml`: if a configuration file is specified without
  an extension, `.yaml` is automatically appended.
- `file/path[option1,option2]`: if a path is followed by brackets with some
  options, `load_config` searches for the options inside that path, and loads
  them in order. If a default config `file/path/path.yaml` is present, its
  contents are also loaded first.
- `file/path[*]`: as a special case of `[options]`, `*` indicates that all
  options (files) inside that path should be loaded. Note that in this case,
  the directory must not contain any non-yaml-loadable files.

For example::

  ["path/a", "path/b[1,2]"]
  # becomes
  ["path/a.yaml", "path/b/b.yaml", "path/b/1.yaml", "path/b/2.yaml"]


The `!include` tag
------------------

The `!include` tag loads the contents of a file, specified as a relative path,
into the file. Note that `!include` can also "merge" its contents into the
including mapping using `<<: !include file.yaml`.

For example::

  parent.yaml:
    x: !include child.yaml
    <<: !include child.yaml

  child.yaml:
    c: 20

  load_config(["parent.yaml"]):
    {"c": 20, "x": {"c": 20}}


The `!expand` tag
-----------------

In order to make listing a subset of files in nested tree structure more
concise and readable, we implement an `!expand` tag that turns a tree-like
structure into a list.

- For each mapping node, each key is treated as a folder, and each value as its
  contents.
- If the value is another mapping, this process is repeated recursively.
- If the value is a sequence, the items in the sequence are treated as files
  within that folder.

For example::

  files.yaml:
    a1:
      b1: ["c11", "c12", "c13"]
      b2: c21
      b3

  load_config(["files.yaml"]):
    ["a1/b1/c11", "a1/b1/c12", "a1/b1/c13", "a1/b2/c21", "a1/b3"]
"""

import json
import os
import re

import yaml
from beartype.typing import Sequence, cast


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
        if k.startswith(":"):
            x[k[1:]] = v
        else:
            if k in x:
                if isinstance(v, dict) and isinstance(x[k], dict):
                    merge_config(x[k], v)
                elif isinstance(v, list) and isinstance(x[k], list):
                    x[k] = x[k] + v
                else:
                    x[k] = v
            else:
                x[k] = v


def parse_paths(configs: Sequence[str]) -> list[str]:
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
            modifier = expandable.group(2)

            if modifier == "*":
                res += [os.path.join(base, m) for m in os.listdir(base)]
            else:
                default = os.path.join(base, os.path.basename(base))
                if os.path.exists(default + ".yaml"):
                    res.append(default)
                for m in modifier.split(','):
                    res.append(os.path.join(base, m))
        else:
            res.append(c)
    return [_add_ext(s) for s in res]


def load_config(*configs: str) -> dict:
    """Load a list of configuration files.

    See :py:mod:`.config` for parsing rules.
    """
    res: dict = {}
    for path in parse_paths(configs):
        with open(path) as f:
            cfg = yaml.load(f, Loader)
        merge_config(res, cfg)
    return res
