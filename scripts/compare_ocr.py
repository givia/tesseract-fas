#!/usr/bin/env python3
"""Pair the text lines two models found on the same pages and show where they differ.

    scripts/compare_ocr.py work/books/zamin stock v1 --sheets 2 --per-sheet 12

Prints agreement statistics, then writes numbered review sheets
(DIR/review/A_vs_B_N.png: line crops) and DIR/review/A_vs_B.json with both
readings of every sampled line, for checking against the image by eye.
"""
import argparse
import collections
import html
import json
import pathlib
import random
import re

from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein

from fa_text import canonical

LINE = re.compile(r"<span class='(ocr_line|ocr_header|ocr_caption|ocr_textfloat)'[^>]*title=\"bbox (\d+) (\d+) (\d+) (\d+);")
WORD = re.compile(r"<span class='ocrx_word'[^>]*>(.*?)</span>", re.S)


def hocr_lines(path):
    text = path.read_text(encoding="utf-8")
    starts = list(LINE.finditer(text))
    lines = []
    for i, m in enumerate(starts):
        chunk = text[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        words = [html.unescape(re.sub("<[^>]+>", "", w)) for w in WORD.findall(chunk)]
        box = tuple(int(g) for g in m.groups()[1:])
        lines.append((box, " ".join(w for w in words if w.strip())))
    return lines


def overlap(a, b):
    """Vertical IoU times horizontal overlap share."""
    v = max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, max(a[3], b[3]) - min(a[1], b[1]))
    h = max(0, min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2] - a[0], b[2] - b[0]))
    return v * h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages_dir")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--sheets", type=int, default=2)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="sample from all paired lines, not only differing ones")
    ap.add_argument("--tag", default="", help="suffix for the review file names")
    opts = ap.parse_args()
    root = pathlib.Path(opts.pages_dir)
    rows, stats = [], collections.Counter()
    for hb in sorted((root / "ocr" / opts.b).glob("*.hocr")):
        la = hocr_lines(root / "ocr" / opts.a / hb.name)
        for box, tb in hocr_lines(hb):
            best = max(la, key=lambda l: overlap(box, l[0]), default=None)
            if best is None or overlap(box, best[0]) < 0.5:
                stats["only_" + opts.b] += 1
                continue
            ca, cb = canonical(best[1]), canonical(tb)
            if len(cb) < 8:
                continue
            stats["lines"] += 1
            d = Levenshtein.distance(ca, cb)
            stats["identical"] += d == 0
            stats["chars"] += len(cb)
            stats["diff_chars"] += d
            if d or opts.all:
                rows.append({"page": hb.stem, "box": box, opts.a: best[1], opts.b: tb, "dist": d})
    print(f"{root.name}: {stats['lines']} paired lines, identical after normalization: "
          f"{stats['identical'] / stats['lines']:.1%}, char difference {stats['diff_chars'] / stats['chars']:.2%}; "
          f"lines only {opts.b} found: {stats['only_' + opts.b]}")

    rng = random.Random(opts.seed)
    sample = rng.sample(rows, min(len(rows), opts.sheets * opts.per_sheet))
    out = root / "review"
    out.mkdir(exist_ok=True)
    for s in range(opts.sheets):
        chunk = sample[s * opts.per_sheet:(s + 1) * opts.per_sheet]
        crops = []
        for r in chunk:
            x0, y0, x1, y1 = r["box"]
            im = Image.open(root / f"{r['page']}.png").convert("L")
            c = im.crop((x0 - 15, y0 - 8, x1 + 15, y1 + 8))
            crops.append(c.resize((max(1, c.width * 60 // c.height), 60)) if c.height > 60 else c)
        if not crops:
            break
        W = max(c.width for c in crops) + 70
        sheet = Image.new("L", (W, sum(c.height + 10 for c in crops)), 255)
        d, y = ImageDraw.Draw(sheet), 0
        for i, c in enumerate(crops):
            sheet.paste(c, (W - c.width, y))
            d.text((5, y + c.height // 2 - 5), f"{s * opts.per_sheet + i + 1}", fill=0)
            d.line((0, y + c.height + 5, W, y + c.height + 5), fill=180)
            y += c.height + 10
        sheet.save(out / f"{opts.a}_vs_{opts.b}{opts.tag}_{s + 1}.png")
    (out / f"{opts.a}_vs_{opts.b}{opts.tag}.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    for i, r in enumerate(sample, 1):
        print(f"\n[{i}] {r['page']}\n  {opts.a:6s}: {r[opts.a]}\n  {opts.b:6s}: {r[opts.b]}")


if __name__ == "__main__":
    main()
