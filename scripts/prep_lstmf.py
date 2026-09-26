#!/usr/bin/env python3
"""Create tesstrain's .box and .lstmf files for a ground-truth dir in parallel.

tesstrain can do this itself, but with tens of thousands of lines GNU make
becomes the bottleneck (it spends its time resolving pattern rules, at ~1k
lines/minute), and its RTL box writer corrupts Persian labels (see
wordstr_box.py). A following `make training` finds these files up to date.

    scripts/prep_lstmf.py work/lines/train
"""
import argparse
import functools
import os
import pathlib
import subprocess
from multiprocessing import Pool

from wordstr_box import wordstr_box

ROOT = pathlib.Path(__file__).resolve().parent.parent


def lstmf_env():
    """`tesseract ... lstm.train` loads the default language (eng) and the stock
    configs; provide both without touching the system tessdata dir."""
    td = ROOT / "work/tessdata_lstmf"
    td.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
    system_tessdata = pathlib.Path(listing.split('"')[1])
    for name, target in [("eng.traineddata", ROOT / "external/tessdata_best/eng.traineddata"),
                         ("configs", system_tessdata / "configs")]:
        if not (td / name).exists():
            (td / name).symlink_to(target)
    return {**os.environ, "TESSDATA_PREFIX": str(td), "OMP_THREAD_LIMIT": "1"}


ENV = lstmf_env()


def prep(png, relabel=False):
    box, lstmf, gt = png.with_suffix(".box"), png.with_suffix(".lstmf"), png.with_suffix(".gt.txt")
    if relabel or not box.exists() or box.stat().st_mtime < gt.stat().st_mtime:
        label = wordstr_box(png, gt.read_text(encoding="utf-8"))
        if not box.exists() or box.read_text(encoding="utf-8") != label:  # unchanged box keeps its lstmf
            box.write_text(label, encoding="utf-8")
    if not lstmf.exists() or lstmf.stat().st_mtime < box.stat().st_mtime:
        proc = subprocess.run(["tesseract", png, png.with_suffix(""), "--psm", "13", "lstm.train"],
                              capture_output=True, text=True, env=ENV)
        if proc.returncode or not lstmf.exists():
            return f"failed {png.name}: {proc.stderr.strip()[-200:]}"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_dir")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--relabel", action="store_true",
                    help="recompute every box after a label-rule change; only changed lines are redone")
    opts = ap.parse_args()
    pngs = sorted(pathlib.Path(opts.gt_dir).rglob("*.png", recurse_symlinks=True))
    with Pool(opts.jobs) as pool:
        results = pool.map(functools.partial(prep, relabel=opts.relabel), pngs, chunksize=64)
    failures = [r for r in results if r != "ok"]
    print(f"{len(pngs) - len(failures)} ok, {len(failures)} failed")
    for f in failures[:10]:
        print(" ", f)


if __name__ == "__main__":
    main()
