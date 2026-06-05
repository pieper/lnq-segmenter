"""Run nnU-Net inference using a downloaded bundle."""
from __future__ import annotations

import os
import shutil

from . import registry, download as _download, cache as _cache


def _ensure_nnunet_layout(bundle, plans_dir):
    """nnUNetPredictor.initialize_from_trained_model_folder reads dataset.json
    and plans.json from the directory it is handed, but publish-model.py
    stages those at bundle root (alongside, not inside, the plans_subdir).
    Plant them where nnU-Net expects without re-minting the bundle."""
    for meta in ("dataset.json", "plans.json"):
        src = os.path.join(bundle, meta)
        dst = os.path.join(plans_dir, meta)
        if not os.path.isfile(src) or os.path.exists(dst):
            continue
        os.makedirs(plans_dir, exist_ok=True)
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def predict(name, input_path, output_path, version=None, folds=None,
            device="cuda", auto_download=True, progress_callback=None):
    """Run a model from the registry on `input_path`, write SEG to `output_path`.

    name              registry name (e.g. 'inguinal-v1')
    input_path        CT volume readable by SimpleITK (.nrrd, .nii.gz, ...)
    output_path       where to write the SEG (same extension as input recommended)
    version           registry version, or None for latest
    folds             subset of ints; defaults to all folds in the registry entry
    device            'cuda' | 'cpu' | 'mps'
    auto_download     if True, fetch missing weights; if False and bundle isn't
                      cached, raise FileNotFoundError instead.
    progress_callback receives structured events ({"event": "predict_start", ...},
                      {"event": "predict_done", "output": ...}, plus all download
                      events when fetching weights).
    """
    entry = registry.get_model(name, version)
    if not _cache.is_complete(entry["name"], entry["version"], entry):
        if not auto_download:
            raise FileNotFoundError(
                f"weights for {entry['name']}@{entry['version']} not cached; "
                f"run `lnq-segmenter download {entry['name']}` first")
        bundle = _download.download(entry, progress_callback=progress_callback)
    else:
        bundle = _cache.bundle_dir(entry["name"], entry["version"])

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    use_folds = tuple(folds) if folds is not None else tuple(entry["folds"])
    plans_dir = os.path.join(bundle, entry["plans_subdir"])
    _ensure_nnunet_layout(bundle, plans_dir)

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
    if progress_callback is not None:
        progress_callback({"event": "predict_start",
                           "model": f"{entry['name']}@{entry['version']}",
                           "input": input_path, "output": output_path,
                           "folds": list(use_folds), "device": device})
    # nnUNetPredictor.predict_from_files appends dataset.json's `file_ending`
    # (e.g. ".nrrd") to whatever filename it is given. Passing "out.nrrd"
    # would land as "out.nrrd.nrrd". Strip the trailing extension before the
    # call so nnU-Net writes directly to `output_path`.
    file_ending = (predictor.dataset_json or {}).get("file_ending", ".nrrd")
    nnunet_out = output_path
    if nnunet_out.lower().endswith(file_ending.lower()):
        nnunet_out = nnunet_out[: -len(file_ending)]
    predictor.predict_from_files(
        [[input_path]],
        [nnunet_out],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    # In case the layout above ever changes, normalize so callers always find
    # the segmentation at `output_path` exactly.
    produced = nnunet_out + file_ending
    if produced != output_path and os.path.isfile(produced):
        os.replace(produced, output_path)
    if progress_callback is not None:
        progress_callback({"event": "predict_done", "output": output_path})
    return output_path
