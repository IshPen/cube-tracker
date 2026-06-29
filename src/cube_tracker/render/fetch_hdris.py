"""Download a varied set of CC0 HDRI environment maps from Poly Haven into assets/hdri/.

HDRIs give the render stage realistic lighting *and* backgrounds in one (domain
randomisation for sim-to-real). They are CC0 (public domain) but, like all assets, are
fetched rather than committed -- see the README. Runs in the project venv:

    python -m cube_tracker.render.fetch_hdris --out-dir assets/hdri --count 28 --resolution 2k
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

_API = "https://api.polyhaven.com"
_HEADERS = {"User-Agent": "cube-tracker/0.0 (research; github.com/IshPen/cube-tracker)"}


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=300) as response:
        dest.write_bytes(response.read())


def fetch_hdris(out_dir: Path, count: int, resolution: str) -> int:
    """Download ``count`` HDRIs spread across Poly Haven's catalogue; skip existing files."""
    ids = sorted(_get_json(f"{_API}/assets?type=hdris"))
    step = max(1, len(ids) // count)
    chosen = ids[::step][:count]
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for position, asset_id in enumerate(chosen, start=1):
        dest = out_dir / f"{asset_id}.hdr"
        if dest.exists():
            continue
        files = _get_json(f"{_API}/files/{asset_id}")["hdri"]
        chosen_res = resolution if resolution in files else next(iter(files))
        _download(files[chosen_res]["hdr"]["url"], dest)
        downloaded += 1
        print(f"[fetch_hdris] {position}/{len(chosen)} {asset_id} ({chosen_res})", flush=True)

    print(
        f"[fetch_hdris] {downloaded} new, {len(chosen) - downloaded} already present -> {out_dir}"
    )
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CC0 HDRIs from Poly Haven.")
    parser.add_argument("--out-dir", default="assets/hdri")
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--resolution", default="2k")
    args = parser.parse_args()
    fetch_hdris(Path(args.out_dir), args.count, args.resolution)


if __name__ == "__main__":
    main()
