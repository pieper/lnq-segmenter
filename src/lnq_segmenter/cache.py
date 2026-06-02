"""On-disk cache for downloaded model bundles.

Cache layout — one dir per (name, version), populated with the canonical
nnU-Net layout so it can be handed straight to nnUNetPredictor:

    <cache_root>/
      inguinal-v1-1.0.0/
        dataset.json
        plans.json
        dataset_fingerprint.json
        card.json
        nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
          fold_0/checkpoint_final.pth
          fold_1/checkpoint_final.pth
          ...
"""
from __future__ import annotations

import os


def cache_root():
    """`~/.cache/lnq-segmenter` (or `$XDG_CACHE_HOME/lnq-segmenter` /
    `$LNQ_SEGMENTER_CACHE` if set)."""
    override = os.environ.get("LNQ_SEGMENTER_CACHE")
    if override:
        return os.path.abspath(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = xdg or os.path.expanduser("~/.cache")
    return os.path.join(base, "lnq-segmenter")


def bundle_dir(name, version):
    """Path to the unpacked bundle for (name, version). Doesn't create it."""
    return os.path.join(cache_root(), f"{name}-{version}")


def expected_files(entry):
    """Files that must exist under bundle_dir for the bundle to be usable.
    Mirrors the assets produced by publish-model.py."""
    plans_subdir = entry.get("plans_subdir") or ""
    out = ["dataset.json", "plans.json", "dataset_fingerprint.json", "card.json"]
    for fold in entry.get("folds") or []:
        out.append(os.path.join(plans_subdir, f"fold_{fold}",
                                "checkpoint_final.pth"))
    return out


def is_complete(name, version, entry):
    """True iff every expected file exists in the bundle dir."""
    root = bundle_dir(name, version)
    return all(os.path.isfile(os.path.join(root, f))
               for f in expected_files(entry))
