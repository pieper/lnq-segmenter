# lnq-segmenter

Lymph node segmentation models for CT, packaged as downloadable nnU-Net weights with a thin Python + CLI wrapper.

Not yet on PyPI — install from this git repo.

## Install

One-liner:

```sh
pip install git+https://github.com/pieper/lnq-segmenter.git
```

Or clone + install (editable, recommended for development):

```sh
git clone https://github.com/pieper/lnq-segmenter.git
cd lnq-segmenter
pip install -r requirements.txt
pip install -e .
```

Both flows pull in `nnunetv2`, `torch`, and `SimpleITK` — they're required, since inference is the point of the package. Weights are downloaded on first use of `predict` (prompted) or eagerly via `lnq-segmenter download`.

## CLI

```sh
lnq-segmenter list                                    # available models
lnq-segmenter info inguinal-v1                        # provenance + asset list
lnq-segmenter download inguinal-v1                    # pre-fetch weights
lnq-segmenter predict inguinal-v1 -i CT.nrrd -o SEG.nrrd
```

`predict` prompts the first time it needs weights (interactive shells only — pipelines and the Slicer extension fetch silently). Use `-y` to skip the prompt.

Weights cache under `~/.cache/lnq-segmenter/<name>-<version>/` and are verified by sha256 from the bundled registry.

## What's in a bundle

Each model release is a set of zip assets — one per fold + a small `meta.zip` with `dataset.json`, `plans.json`, `dataset_fingerprint.json`, and `card.json`. Unzipped together, they form the canonical nnU-Net layout that `nnUNetPredictor.initialize_from_trained_model_folder()` accepts directly — no extra glue.

## Releasing a new model

See `SlicerLNQ-Chronicler/bin/publish-model.py` — it produces the per-fold zips and a registry-entry JSON snippet to append to `_registry.json`.
