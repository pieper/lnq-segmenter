"""Read the bundled _registry.json and expose query helpers.

Registry shape:
    {
      "schema": "lnq-segmenter-registry/1",
      "models": [
        { ...one entry per (name, version) — produced by publish-model.py },
        ...
      ]
    }
"""
from __future__ import annotations

import json
import os
from importlib import resources


def _load():
    with resources.files(__package__).joinpath("_registry.json").open() as f:
        return json.load(f)


def list_models():
    """All entries in registry order. Each entry is the dict produced by
    publish-model.py's make_registry_entry()."""
    return list(_load().get("models") or [])


def _semver_key(v):
    parts = []
    for chunk in (v or "0").split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def latest_version(name):
    """Highest semver string for `name`, or None if no entries match."""
    versions = [m["version"] for m in list_models() if m["name"] == name]
    if not versions:
        return None
    return max(versions, key=_semver_key)


def get_model(name, version=None):
    """Return the registry entry for (name, version). If version is None,
    returns the latest. Raises KeyError if no match."""
    if version is None:
        version = latest_version(name)
        if version is None:
            raise KeyError(f"no model named {name!r} in registry")
    for m in list_models():
        if m["name"] == name and m["version"] == version:
            return m
    raise KeyError(f"no model {name}@{version} in registry")


def parse_name_at_version(arg):
    """Split 'name' or 'name@version' into (name, version_or_None)."""
    name, sep, version = arg.partition("@")
    return name, (version or None)
