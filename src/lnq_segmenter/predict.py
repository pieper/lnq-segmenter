"""Run nnU-Net inference using a downloaded bundle."""
from __future__ import annotations

import os

from . import registry, download as _download, cache as _cache


def predict(name, input_path, output_path, version=None, folds=None,
            device="cuda", auto_download=True):
    """Run a model from the registry on `input_path`, write SEG to `output_path`.

    name           registry name (e.g. 'inguinal-v1')
    input_path     CT volume readable by SimpleITK (.nrrd, .nii.gz, ...)
    output_path    where to write the SEG (same extension as input recommended)
    version        registry version, or None for latest
    folds          subset of ints; defaults to all folds in the registry entry
    device         'cuda' | 'cpu' | 'mps'
    auto_download  if True, fetch missing weights; if False and bundle isn't
                   cached, raise FileNotFoundError instead.
    """
    entry = registry.get_model(name, version)
    if not _cache.is_complete(entry["name"], entry["version"], entry):
        if not auto_download:
            raise FileNotFoundError(
                f"weights for {entry['name']}@{entry['version']} not cached; "
                f"run `lnq-segmenter download {entry['name']}` first")
        bundle = _download.download(entry)
    else:
        bundle = _cache.bundle_dir(entry["name"], entry["version"])

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    use_folds = tuple(folds) if folds is not None else tuple(entry["folds"])
    plans_dir = os.path.join(bundle, entry["plans_subdir"])

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False, verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        plans_dir, use_folds=use_folds, checkpoint_name="checkpoint_final.pth")

    # predict_from_files wants list-of-list for multi-channel inputs; CT is
    # single-channel, so each item is a 1-element list.
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    predictor.predict_from_files(
        [[input_path]],
        [output_path],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    return output_path
