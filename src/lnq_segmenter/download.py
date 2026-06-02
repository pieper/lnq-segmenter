"""Fetch + verify weights assets from the URL template in the registry entry.

Assets are zipped subsets of the canonical nnU-Net layout (one fold each,
plus a small meta.zip). Each asset is downloaded to a temp file, sha256-
verified, then unzipped into the bundle dir. Re-running is idempotent —
assets that match the expected sha are skipped."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile

from . import cache as _cache


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _resolve_url(entry, asset):
    return entry["weights_url_template"].format(
        name=entry["name"], version=entry["version"],
        filename=asset["filename"])


def _fetch(url, dst_path, expected_sha256, expected_size, progress=True):
    """Download `url` to `dst_path` (atomic via temp file), verify sha256.
    Progress bar only when stderr is a TTY — non-interactive callers get a
    single completion line per asset instead."""
    tmp_dir = os.path.dirname(dst_path)
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dl-", dir=tmp_dir)
    os.close(fd)
    show_bar = progress and sys.stderr.isatty()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, \
                open(tmp, "wb") as out:
            total = expected_size or 0
            done = 0
            chunk = 1 << 20
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if show_bar and total:
                    pct = 100.0 * done / total
                    sys.stderr.write(
                        f"\r  {os.path.basename(dst_path):44s}  "
                        f"{done / 1e6:7.1f} / {total / 1e6:7.1f} MB  "
                        f"{pct:5.1f}%")
                    sys.stderr.flush()
        if show_bar and total:
            sys.stderr.write("\n")
        elif progress:
            sys.stderr.write(
                f"  fetched {os.path.basename(dst_path)} "
                f"({done / 1e6:.1f} MB)\n")
        actual = _sha256_file(tmp)
        if actual != expected_sha256:
            raise RuntimeError(
                f"sha256 mismatch for {url}: expected {expected_sha256}, "
                f"got {actual}")
        os.replace(tmp, dst_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _unzip(zip_path, dst_dir):
    """Extract all members of `zip_path` into `dst_dir`. Refuses paths that
    would escape (defense against ZipSlip)."""
    dst_dir = os.path.abspath(dst_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            out_path = os.path.abspath(os.path.join(dst_dir, info.filename))
            if not out_path.startswith(dst_dir + os.sep) and out_path != dst_dir:
                raise RuntimeError(f"refusing path outside bundle: {info.filename}")
            if info.is_dir():
                os.makedirs(out_path, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)


def download(entry, progress=True):
    """Ensure every asset listed in `entry['weights_assets']` is present in
    the bundle dir. Returns the bundle dir path."""
    name, version = entry["name"], entry["version"]
    bundle = _cache.bundle_dir(name, version)
    os.makedirs(bundle, exist_ok=True)
    dl_dir = os.path.join(bundle, ".downloads")
    os.makedirs(dl_dir, exist_ok=True)

    expected = _cache.expected_files(entry)
    if all(os.path.isfile(os.path.join(bundle, f)) for f in expected):
        if progress:
            sys.stderr.write(f"[lnq-segmenter] {name}@{version} already cached\n")
        return bundle

    for asset in entry["weights_assets"]:
        zip_path = os.path.join(dl_dir, asset["filename"])
        need = True
        if os.path.isfile(zip_path) and os.path.getsize(zip_path) == \
                asset["size_bytes"]:
            if _sha256_file(zip_path) == asset["sha256"]:
                need = False
        if need:
            url = _resolve_url(entry, asset)
            if progress:
                sys.stderr.write(f"[lnq-segmenter] fetch {url}\n")
            _fetch(url, zip_path, asset["sha256"], asset["size_bytes"],
                   progress=progress)
        if progress:
            sys.stderr.write(f"[lnq-segmenter] unzip {asset['filename']}\n")
        _unzip(zip_path, bundle)
    return bundle
