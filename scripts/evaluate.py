#!/usr/bin/env python3
"""Evaluate Tesseract models on the real-world page benchmark or a line set.

    scripts/evaluate.py --model stock_best:external/tessdata_best:fas \
                        --model mine:work/models:fas_custom:"-c textord_noise_rejrows=0"
    scripts/evaluate.py --set work/lines/holdout --model ...

A model spec is LABEL:TESSDATA_DIR:LANG[:EXTRA_TESSERACT_ARGS]. --set is
"bench" (default) or a directory of NAME.png + NAME.gt.txt line pairs, which are
recognized as single lines (--psm 7). OCR output is cached in
work/eval/SET/LABEL/, so re-running only scores; pass --rerun to redo OCR.

Metrics (micro-averaged over all pages or lines):
  CER strict   edit distance after canonical() only (orthography counts)
  CER relaxed  after relaxed() (digit/punctuation script, ZWNJ, diacritics folded)
  CER no spaces  relaxed, with all spaces removed (robust to word-spacing conventions)
  WER relaxed  word-level edit distance on relaxed text
  Word recall  share of ground-truth words found anywhere on the page (order-free,
               so it is not distorted by reading-order differences on forms)
"""
import argparse
import collections
import json
import os
import pathlib
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from rapidfuzz.distance import Levenshtein

from fa_text import canonical, relaxed

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "external/benchmark"

# Characters that should never appear in canonical Persian output.
NON_CANONICAL = {"ي": "Arabic yeh", "ك": "Arabic kaf", "ى": "alef maksura", "٩": "Arabic-Indic 9",
                 "‎": "LRM", "‏": "RLM"}


def readable_by_leptonica(image):
    """Leptonica fails on some GIFs in the benchmark; hand it a cached PNG instead."""
    if image.suffix.lower() != ".gif":
        return image
    png = ROOT / "work/bench_png" / (image.stem + ".png")
    if not png.exists():
        png.parent.mkdir(parents=True, exist_ok=True)
        Image.open(image).convert("RGB").save(png)
    return png


def upscaled(image, min_width):
    """Many benchmark pages are small web images whose text is too small for
    Tesseract's layout analysis; optionally upscale them (Lanczos), cached."""
    im = Image.open(image)
    if not min_width or im.width >= min_width:
        return image
    out = ROOT / f"work/bench_w{min_width}" / (image.stem + ".png")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        scale = min_width / im.width
        im.convert("RGB").resize((min_width, round(im.height * scale)), Image.LANCZOS).save(out)
    return out


def load_benchmark(page_type=None, min_width=None):
    """page_type: "printed", "handwritten" or "mixed" (see data/benchmark_labels.tsv), None = all."""
    rows = [json.loads(line) for line in open(BENCH / "ocr_eval.jsonl", encoding="utf-8-sig")]
    labels = dict(line.split("\t")[:2] for line in open(ROOT / "data/benchmark_labels.tsv")
                  if not line.startswith(("#", "id\t")))
    return [(r["id"], upscaled(readable_by_leptonica(BENCH / r["image"]), min_width), r["text"])
            for r in rows if page_type in (None, labels[r["id"]])]


def load_lines(directory):
    """Line images with a transcription; images without one (not yet corrected) are left out."""
    return [(png.stem, png, png.with_suffix(".gt.txt").read_text(encoding="utf-8"))
            for png in sorted(pathlib.Path(directory).glob("*.png")) if png.with_suffix(".gt.txt").exists()]


def run_ocr(spec, set_name, pages, rerun, jobs, line_mode):
    label, tessdata, lang, *extra = spec.split(":", 3)
    out_dir = ROOT / "work/eval" / set_name / label
    out_dir.mkdir(parents=True, exist_ok=True)
    args = shlex.split(extra[0]) if extra else []
    if line_mode and "--psm" not in args:
        args += ["--psm", "7"]
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}

    def one(page):
        page_id, image, _ = page
        out = out_dir / f"{page_id}.txt"
        if rerun or not out.exists():
            proc = subprocess.run(
                ["tesseract", str(image), "-", "--tessdata-dir", str(ROOT / tessdata), "-l", lang, *args],
                capture_output=True, text=True, env=env)
            if proc.returncode < 0:  # killed by a signal (e.g. SIGFPE): score as empty output
                (out_dir / f"{page_id}.crash").write_text(f"signal {-proc.returncode}\n")
            elif proc.returncode:
                raise RuntimeError(f"{label} {image.name}: {proc.stderr.strip()}")
            out.write_text(proc.stdout, encoding="utf-8")
        return out.read_text(encoding="utf-8")

    with ThreadPoolExecutor(jobs) as pool:
        outputs = list(pool.map(one, pages))
    crashes = sum((out_dir / f"{p[0]}.crash").exists() for p in pages)
    if crashes:
        print(f"{label}: tesseract crashed on {crashes} items (scored as empty output)")
    return label, outputs


def score(pages, outputs):
    totals = collections.Counter()
    for (_, _, gt), raw in zip(pages, outputs):
        g_strict, o_strict = canonical(gt), canonical(raw)
        g_rel, o_rel = relaxed(gt), relaxed(raw)
        totals["cer_strict"] += Levenshtein.distance(g_strict, o_strict)
        totals["chars_strict"] += len(g_strict)
        totals["cer_relaxed"] += Levenshtein.distance(g_rel, o_rel)
        g_ns, o_ns = "".join(g_rel.split()), "".join(o_rel.split())
        totals["cer_nospace"] += Levenshtein.distance(g_ns, o_ns)
        totals["chars_nospace"] += len(g_ns)
        totals["chars_relaxed"] += len(g_rel)
        g_words, o_words = g_rel.split(), o_rel.split()
        totals["wer"] += Levenshtein.distance(g_words, o_words)
        totals["words"] += len(g_words)
        totals["found"] += sum((collections.Counter(g_words) & collections.Counter(o_words)).values())
        totals["out_chars"] += len(raw)
        for ch in NON_CANONICAL:
            totals[ch] += raw.count(ch)
    return {
        "CER strict": totals["cer_strict"] / totals["chars_strict"],
        "CER relaxed": totals["cer_relaxed"] / totals["chars_relaxed"],
        "CER no spaces": totals["cer_nospace"] / totals["chars_nospace"],
        "WER relaxed": totals["wer"] / totals["words"],
        "Word recall": totals["found"] / totals["words"],
        "non-canonical chars": {name: totals[ch] for ch, name in NON_CANONICAL.items() if totals[ch]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="LABEL:TESSDATA_DIR:LANG[:ARGS]")
    ap.add_argument("--set", default="bench", help='"bench" or a directory of NAME.png + NAME.gt.txt pairs')
    ap.add_argument("--pages", action="store_true", help="the --set directory holds full pages, not lines")
    ap.add_argument("--page-type", choices=["printed", "handwritten", "mixed"],
                    help="bench only: restrict to one page type (data/benchmark_labels.tsv)")
    ap.add_argument("--min-width", type=int, help="bench only: upscale narrower pages to this width")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, help="only the first N pages (quick checks)")
    opts = ap.parse_args()

    set_name = pathlib.Path(opts.set).name
    if opts.set == "bench" and opts.min_width:
        set_name = f"bench_w{opts.min_width}"
    pages = (load_benchmark(opts.page_type, opts.min_width) if opts.set == "bench" else load_lines(opts.set))[: opts.limit]
    results = {}
    for spec in opts.model:
        label, outputs = run_ocr(spec, set_name, pages, opts.rerun, opts.jobs,
                                 line_mode=opts.set != "bench" and not opts.pages)
        results[label] = score(pages, outputs)

    print(f"{set_name}{' ' + opts.page_type if opts.page_type else ''}: {len(pages)} items, {sum(len(canonical(p[2])) for p in pages)} ground-truth chars\n")
    cols = ["CER strict", "CER relaxed", "CER no spaces", "WER relaxed", "Word recall"]
    print("| model | " + " | ".join(cols) + " | non-canonical chars emitted |")
    print("|---" * (len(cols) + 2) + "|")
    for label, m in results.items():
        bad = ", ".join(f"{k} {v}" for k, v in m["non-canonical chars"].items()) or "none"
        print(f"| {label} | " + " | ".join(f"{m[c]:.2%}" for c in cols) + f" | {bad} |")
    (ROOT / "work/eval" / set_name / f"results{'-' + opts.page_type if opts.page_type else ''}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
