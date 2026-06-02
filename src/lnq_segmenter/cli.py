"""lnq-segmenter CLI: list / info / download / predict."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import registry, download as _download, cache as _cache, __version__


def _resolve_entry(spec):
    name, version = registry.parse_name_at_version(spec)
    return registry.get_model(name, version)


def cmd_list(args):
    rows = registry.list_models()
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if not rows:
        print("(no models in registry)")
        return 0
    width = max(len(m["name"]) for m in rows)
    for m in rows:
        size = sum(a["size_bytes"] for a in m.get("weights_assets") or [])
        cached = "  " if not _cache.is_complete(m["name"], m["version"], m) else "✓ "
        print(f"{cached}{m['name']:{width}s}  {m['version']:>8s}  "
              f"{size / 1e6:7.0f} MB   {m.get('display_name', '')}")
    return 0


def cmd_info(args):
    entry = _resolve_entry(args.spec)
    if args.json:
        json.dump(entry, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(f"{entry['name']}@{entry['version']}  — {entry.get('display_name', '')}")
    print(f"  license:           {entry.get('license')}")
    print(f"  task / modality:   {entry.get('task')} / {entry.get('modality')}")
    print(f"  trainer / plans:   {entry.get('trainer_class')} / "
          f"{entry.get('plans_identifier')} / {entry.get('configuration')}")
    print(f"  folds:             {entry.get('folds')}")
    print(f"  labels:            {entry.get('label_map')}")
    tp = entry.get("training_provenance") or {}
    if tp:
        print(f"  trained on:        {tp.get('training_set_size')} cases")
        print(f"  finished:          {tp.get('training_finished_at')}")
        print(f"  model_generation:  {tp.get('model_generation_id')}")
        if tp.get("chronicle_url"):
            print(f"  chronicle:         {tp['chronicle_url']}")
    if entry.get("recommended_use"):
        print(f"  recommended use:   {entry['recommended_use']}")
    print("  assets:")
    for a in entry.get("weights_assets") or []:
        print(f"    {a['role']:8s}  {a['size_bytes'] / 1e6:7.1f} MB  "
              f"sha256={a['sha256'][:12]}…  {a['filename']}")
    bundle = _cache.bundle_dir(entry["name"], entry["version"])
    if _cache.is_complete(entry["name"], entry["version"], entry):
        print(f"  cached at:         {bundle}")
    else:
        print(f"  cache target:      {bundle} (not downloaded)")
    return 0


def cmd_download(args):
    entry = _resolve_entry(args.spec)
    bundle = _download.download(entry, progress=not args.quiet)
    print(bundle)
    return 0


def _prompt_for_download(entry):
    """Prompt the user before fetching weights on first predict. Auto-yes
    when stdin isn't a TTY (so scripts and the Slicer extension don't hang)."""
    if _cache.is_complete(entry["name"], entry["version"], entry):
        return True
    total_mb = sum(a["size_bytes"] for a in entry["weights_assets"]) / 1e6
    bundle = _cache.bundle_dir(entry["name"], entry["version"])
    print(f"Model {entry['name']}@{entry['version']} is not cached.")
    print(f"  {len(entry['weights_assets'])} assets, {total_mb:.0f} MB total")
    print(f"  destination: {bundle}")
    if not sys.stdin.isatty():
        print("  (non-interactive stdin — proceeding)")
        return True
    answer = input("Download now? [Y/n]: ").strip().lower()
    return answer in ("", "y", "yes")


def cmd_predict(args):
    from . import predict as _predict
    entry = _resolve_entry(args.spec)
    if not args.yes and not _prompt_for_download(entry):
        print("aborted.", file=sys.stderr)
        return 1
    out = _predict.predict(
        entry["name"], args.input, args.output,
        version=entry["version"],
        folds=args.folds,
        device=args.device,
    )
    print(out)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="lnq-segmenter",
        description="Lymph node segmentation models — CLI + registry.")
    ap.add_argument("--version", action="version",
                    version=f"lnq-segmenter {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="List models in the bundled registry.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="Show details for a model.")
    p.add_argument("spec", help="name[@version]")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("download", help="Fetch + verify all model assets.")
    p.add_argument("spec", help="name[@version]")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("predict",
                       help="Run inference on a CT volume. Requires [predict] extra.")
    p.add_argument("spec", help="name[@version]")
    p.add_argument("--input", "-i", required=True, help="Input CT path.")
    p.add_argument("--output", "-o", required=True, help="Output SEG path.")
    p.add_argument("--folds", type=int, nargs="*", default=None,
                   help="Folds to use (default: all from registry).")
    p.add_argument("--device", default="cuda",
                   choices=("cuda", "cpu", "mps"))
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the download-on-first-run prompt.")
    p.set_defaults(func=cmd_predict)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as e:
        print(f"lnq-segmenter: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
