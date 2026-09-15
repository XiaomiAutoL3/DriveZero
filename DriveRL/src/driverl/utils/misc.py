__all__ = [
    "seed_everything",
    "RecursiveLoader",
    "apply_overrides",
    "apply_extends",
    "deep_merge_dict",
]
import os

import yaml


def seed_everything(seed, torch_deterministic):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    import torch

    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


class RecursiveLoader(yaml.SafeLoader):
    """Recursive YAML loader."""

    # Custom PyYAML loader, refactor from: https://stackoverflow.com/a/9577670
    def __init__(self, stream):
        """Init recursive loader."""
        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def _include(self, node):
        if isinstance(node.value, list):
            filename = "".join(self.construct_sequence(node))
        else:
            filename = self.construct_scalar(node)

        with open(os.path.join(self._root, filename), "r") as f:
            return yaml.load(f, Loader=RecursiveLoader)

    def _concat(self, node):
        seq = self.construct_sequence(node)
        return "".join(seq)

    def _flatten(self, node):
        # flatten yaml sequence node non-recursively
        res = []
        for value in node.value:
            if isinstance(value, yaml.ScalarNode):
                res.append(value)
            elif isinstance(value, yaml.SequenceNode):
                res += value.value
            else:
                raise NotImplementedError(
                    f"Yaml custom flatten doesn't support type {type(value)}"
                )
        return [self.construct_object(child, deep=False) for child in res]

    def _extends(self, node):
        # Loads a parent yaml so the child can override fields via deep merge.
        # The actual merge happens after the whole document is loaded, in
        # ``_apply_extends``; here we just return the parent dict tagged with
        # a sentinel key so we can detect it at the top level.
        filename = self.construct_scalar(node)
        with open(os.path.join(self._root, filename), "r") as f:
            return yaml.load(f, Loader=RecursiveLoader)


RecursiveLoader.add_constructor("!include", RecursiveLoader._include)
RecursiveLoader.add_constructor("!concat", RecursiveLoader._concat)
RecursiveLoader.add_constructor("!flatten", RecursiveLoader._flatten)
RecursiveLoader.add_constructor("!extends", RecursiveLoader._extends)


_EXTENDS_KEY = "__extends__"


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``. Dicts merge key-wise;
    every other type (list, scalar) is replaced wholesale by ``override``.
    Returns a new dict; inputs are not mutated.
    """
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def apply_extends(config_data: dict) -> dict:
    """Resolve the top-level ``__extends__`` field by deep-merging the parent
    config under the current one (current wins). Supports chained inheritance
    because the parent has already been resolved by the time we see it (its
    own ``__extends__`` was unwrapped during YAML loading recursion).
    """
    if not isinstance(config_data, dict) or _EXTENDS_KEY not in config_data:
        return config_data
    parent = config_data.pop(_EXTENDS_KEY)
    if not isinstance(parent, dict):
        raise TypeError(
            f"`{_EXTENDS_KEY}` must reference a yaml mapping, got {type(parent)}"
        )
    parent = apply_extends(parent)
    return deep_merge_dict(parent, config_data)


def apply_overrides(config_data: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides (key.subkey=value) to a nested dict in-place.

    Missing intermediate keys are created automatically as dictionaries so
    overrides can target newly introduced config fields even when they are not
    present in an older backed-up ``config.yaml``.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        path, raw_value = override.split("=", 1)
        keys = path.split(".")
        value = yaml.safe_load(raw_value)

        cursor = config_data
        for key in keys[:-1]:
            if key not in cursor:
                cursor[key] = {}
            elif not isinstance(cursor[key], dict):
                # If the existing value is not a dict, replace it so nested
                # overrides can still be applied to legacy configs.
                cursor[key] = {}
            cursor = cursor[key]

        leaf = keys[-1]
        cursor[leaf] = value
    return config_data
