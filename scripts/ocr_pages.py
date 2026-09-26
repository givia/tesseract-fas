#!/usr/bin/env python3
"""OCR a directory of page images with several models; keep text and hOCR.

    scripts/ocr_pages.py work/books/zamin --model stock:external/tessdata_best:fas \
        --model v1:work/data:fas_v1:"-c tessedit_right_to_left=1"

Writes DIR/ocr/LABEL/PAGE.txt and PAGE.hocr; existing outputs are kept.
Model specs are as in evaluate.py.
"""
import argparse
import os
import pathlib
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages_dir")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--tesseract", default="tesseract")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    opts = ap.parse_args()
    pages = sorted(pathlib.Path(opts.pages_dir).glob("*.png"))
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    local = ROOT / "work/tess-local"
    env["LD_LIBRARY_PATH"] = f"{local}/lib:{local}/lib64:" + env.get("LD_LIBRARY_PATH", "")

    tasks = []
    for spec in opts.model:
        label, tessdata, lang, *extra = spec.split(":", 3)
        out_dir = pathlib.Path(opts.pages_dir) / "ocr" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        args = shlex.split(extra[0]) if extra else []
        for page in pages:
            base = out_dir / page.stem
            if not base.with_suffix(".hocr").exists():
                tasks.append([opts.tesseract, str(page), str(base), "--tessdata-dir", str(ROOT / tessdata),
                              "-l", lang, *args, "-c", "tessedit_create_hocr=1", "-c", "tessedit_create_txt=1"])

    def run(cmd):
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return proc.returncode

    with ThreadPoolExecutor(opts.jobs) as pool:
        codes = list(pool.map(run, tasks))
    print(f"{len(tasks)} runs, {sum(c != 0 for c in codes)} failed")


if __name__ == "__main__":
    main()
