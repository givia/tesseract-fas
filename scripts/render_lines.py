#!/usr/bin/env python3
"""Render corpus lines into tesstrain line pairs (NAME.png + NAME.gt.txt).

Each line is rendered with text2image in one font that fully covers its
characters (text2image otherwise silently falls back to another system font or
drops words), cropped, and degraded to look like real scans and photos.

The rendered text differs from the label in ways print does:
  * kashida (tatweel) stretching inside words, dropped from the label;
  * dialogue dashes typeset as a standalone tatweel ("ـ چای"), labelled "-";
  * narrow or wide word spaces (thin/en spaces), labelled " ". Persian
    books set word spaces barely wider than a ZWNJ gap (hair spaces are too
    narrow: they taught the model to split words at every letter gap);
  * a standalone tatweel as a range dash between numbers ("۱۳۶۳ـ۱۳۶۴"), labelled "-";
  * some sentences turned into questions or exclamations: Wikipedia has almost
    no "؟", which the model then misread as "؛";
with --fiction also
  * ellipses as fiction uses them ("خواهرا... برادرا...", "!...", "ـ... ");
  * a space before punctuation, as in older typesetting ("کرد . ولی");
  * Arabic-Indic digit shapes (١٩٤١), labelled with Persian digits.
The label is canonical(rendered text), so it always describes the image; a
dialogue dash is always labelled "- ", with or without a visible gap.

--diacritics P keeps vowel marks in a share P of lines and strips the rest
(tanwin-fath and hamza, common in modern text, are always kept). Google's
training text, used for rehearsal, has ~80 marks per 1,000 characters against
~1 in modern books, which made the model add vowel marks that are not there.

Many Iranian fonts (all "B" fonts) draw ASCII digits with Persian digit shapes;
for those the label uses Persian digits, which is what the image shows.

--degrade mix  (default) resolution loss, blur, noise, contrast, JPEG, skew,
               and binarized scan looks (thick/thin strokes, specks);
--degrade scan always the binarized scan look, often with dense pepper noise;
--degrade t2i  text2image's own photocopier degradation at 300 dpi, as used
               for the stock model; for rehearsal data.

    scripts/render_lines.py work/corpus/lines.txt --fonts train --out work/lines/train --count 50000
    scripts/render_lines.py work/corpus/lines.txt --fonts holdout --out work/lines/holdout \\
        --skip 194000 --count 3000
"""
import argparse
import collections
import io
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from fa_text import PERSIAN_DIGITS, canonical

ROOT = pathlib.Path(__file__).resolve().parent.parent
F = ROOT / "external/fonts"
SYS = pathlib.Path("/usr/share/fonts")

FONTS = {
    # Held-out families are never trained on, to measure generalization to unseen type.
    "holdout": [F / "parastoo/Parastoo.ttf", F / "parastoo/Parastoo-Bold.ttf",
                F / "estedad/Estedad-Regular.ttf", F / "estedad/Estedad-Bold.ttf",
                F / "farsiweb/homa.ttf", F / "bfonts/BYas.ttf"],
    "holdout_book": [F / "bfonts/BYas.ttf"],
    "train": [
        *(F / f"vazirmatn/Vazirmatn-{w}.ttf" for w in ("Regular", "Bold", "Light", "Medium")),
        *(F / f"sahel/{n}.ttf" for n in ("Sahel", "Sahel-Bold", "Sahel-Light")),
        *(F / f"samim/{n}.ttf" for n in ("Samim", "Samim-Bold", "Samim-Medium")),
        *(F / f"shabnam/{n}.ttf" for n in ("Shabnam", "Shabnam-Bold", "Shabnam-Light")),
        F / "tanha/Tanha.ttf", F / "gandom/Gandom.ttf", F / "nahid/Nahid.ttf",
        F / "farsiweb/nazli.ttf", F / "farsiweb/nazlib.ttf", F / "farsiweb/titr.ttf",
        *(F / f"googlefonts/{n}.ttf" for n in ("Amiri-Regular", "Amiri-Bold", "ScheherazadeNew-Regular",
                                              "ScheherazadeNew-Bold", "Mirza-Regular", "Mirza-Bold")),
        SYS / "noto/NotoNaskhArabic-Regular.ttf", SYS / "noto/NotoNaskhArabic-Bold.ttf",
        SYS / "noto/NotoSansArabic-Regular.ttf", SYS / "noto/NotoSansArabic-Bold.ttf",
        SYS / "TTF/DejaVuSans.ttf", SYS / "TTF/DejaVuSans-Bold.ttf",
    ],
    # Classic Iranian book and office faces. B Homa is left out: it is the same
    # design as the held-out FarsiWeb Homa.
    "book": [F / f"bfonts/{n}.ttf" for n in ("BLotus", "BMitra", "BNazanin", "BZar", "BBadr", "BYagut",
                                              "BRoya", "BCompset", "BTraffic", "BYekan", "BTitrBold",
                                              "BKoodakBold")],
}

TATWEEL = "ـ"
DUAL_JOINING = set("بپتثجچحخسشصضطظعغفقکگلمنهیئ")
PERSIAN_LETTERS = DUAL_JOINING | set("اآدذرزژو")
NARROW_SPACES = [" ", " ", " "]  # thin, six-per-em, four-per-em
WIDE_SPACE = " "  # en space
IGNORED_IN_BOXES = {" ", "\t", "‌", TATWEEL, WIDE_SPACE, *NARROW_SPACES}
TO_PERSIAN_DIGITS = str.maketrans("0123456789", PERSIAN_DIGITS)
VOWEL_MARKS = re.compile("[\u064c-\u0652\u0670]")  # all harakat but fathatan, plus dagger alef
DIGIT_RANGE = re.compile(f"(?<=[0-9{PERSIAN_DIGITS}])-(?=[0-9{PERSIAN_DIGITS}])")
DIALOGUE_DASH = re.compile(r"^-(?=[^\s.])")
RENDER_DPI = 600  # rendered bilevel, downscaled later for natural anti-aliasing


def text2image(text, font_name, tmp, fc_dir, ptsize, dpi, extra=()):
    """Render one line; return (page image, rendered chars from the box file) or None."""
    src = pathlib.Path(tmp) / "line.txt"
    src.write_text(text + "\n", encoding="utf-8")
    proc = subprocess.run(
        ["text2image", f"--text={src}", f"--outputbase={tmp}/out", f"--font={font_name}",
         f"--fonts_dir={ROOT / 'work/fonts'}", f"--fontconfig_tmpdir={fc_dir}",
         f"--ptsize={ptsize}", f"--resolution={dpi}", f"--xsize={dpi * 14}", f"--ysize={dpi}",
         f"--margin={dpi // 10}", "--leading=0", *extra],
        capture_output=True, text=True)
    box = pathlib.Path(tmp) / "out.box"
    if proc.returncode or not box.exists():
        return None
    rendered = collections.Counter(c for l in box.read_text(encoding="utf-8").splitlines()
                                   for c in l.rsplit(" ", 5)[0] if c not in IGNORED_IN_BOXES)
    return Image.open(pathlib.Path(tmp) / "out.tif").convert("L"), rendered


def font_info(path):
    """(pango font name, covered code points, (draws ASCII digits as Persian, has Arabic-Indic digits))."""
    fmt = "%{family[0]}\t%{style[0]}\t%{charset}"
    family, style, charset = subprocess.run(["fc-query", f"--format={fmt}", str(path)],
                                            capture_output=True, text=True, check=True).stdout.split("\t")
    covered = set()
    for rng in charset.split():
        lo, _, hi = rng.partition("-")
        covered.update(range(int(lo, 16), int(hi or lo, 16) + 1))
    name = family if style in ("Regular", "Book", "Normal") else f"{family} {style}"
    arabic_indic = all(0x660 + d in covered for d in range(10))
    return name, covered, (persian_ascii_digits(path, name), arabic_indic)


def persian_ascii_digits(path, name):
    """Render 0-9 and ۰-۹; identical images mean the font draws ASCII digits as Persian. Cached."""
    cache_file = ROOT / "work/fonts/digit_check.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = f"{path}:{path.stat().st_size}"
    if key not in cache:
        with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as tmp:
            fc_dir = pathlib.Path(tmp) / "fc"
            fc_dir.mkdir()
            crops = []
            for digits in ("0123456789", PERSIAN_DIGITS):
                page, _ = text2image(digits, name, tmp, fc_dir, 24, 300, ["--degrade_image=false",
                                                                          "--rotate_image=false"])
                crops.append(np.asarray(page.crop(Image.eval(page, lambda p: 255 - p).getbbox())))
        a, b = crops
        cache[key] = bool(a.shape == b.shape and (a != b).mean() < 0.02)
        cache_file.write_text(json.dumps(cache, indent=1))
    return cache[key]


def add_kashida(text, rng):
    chars = list(text)
    spots = [i for i in range(len(chars) - 1) if chars[i] in DUAL_JOINING and chars[i + 1] in PERSIAN_LETTERS]
    # Justified book lines stretch many words, mostly by a little.
    for i in sorted(rng.sample(spots, min(len(spots), rng.randint(1, 6))), reverse=True):
        chars.insert(i + 1, TATWEEL * rng.choice([1, 1, 2, 2, 3, 4, 6]))
    return "".join(chars)


def vary_spaces(text, rng):
    if rng.random() < 0.8:  # tight book setting: mostly narrow spaces
        return "".join(rng.choice(NARROW_SPACES + [" "]) if c == " " else c for c in text)
    return text.replace(" ", WIDE_SPACE)  # loosely justified line


ARABIC_INDIC = str.maketrans(PERSIAN_DIGITS, "٠١٢٣٤٥٦٧٨٩")


def fiction_punctuation(text, rng):
    words = text.split(" ")
    if rng.random() < 0.6 and len(words) > 2:  # ellipsis after a word inside the line
        i = rng.randrange(len(words) - 1)
        words[i] += rng.choice(["...", "...", "!...", "؟..."])
    else:  # ellipsis at the end
        words[-1] = words[-1].rstrip(".،") + rng.choice(["...", "...", "!...", "؟...", "!"])
    return " ".join(words)


def questions(text, rng):
    """Turn a sentence end into "؟" (or "!"): a visual drill, the text need not make sense."""
    mark = "؟" if rng.random() < 0.8 else "!"
    ends = [i for i, c in enumerate(text) if c == "." and text[i - 1:i] != "."  # not an ellipsis
            and (i == len(text) - 1 or text[i + 1] == " ")]
    if ends:
        i = rng.choice(ends)
        return text[:i] + mark + text[i + 1:]
    return text.rstrip("،؛:") + mark


def spaced_punctuation(text):
    for mark in ".،؟!:؛":
        text = text.replace(mark, " " + mark)
    return text.replace("  ", " ")


def scan_look(img, rng, scale, pepper):
    """Binarized office/book scan: stroke weight changes, threshold, specks.

    pepper: chance of dense 1-2 px noise, as on cheap book scans. Some of it lands
    where a letter's dot would go (ر/ز, ح/خ); it is smaller than a real dot.
    """
    if rng.random() < 0.5:  # at render resolution: bolder (min) or thinner (max) strokes
        # Thinning at low resolution erases thin faces completely.
        thin = scale >= 0.35 and rng.random() < 0.5
        img = img.filter(ImageFilter.MaxFilter(3) if thin else ImageFilter.MinFilter(rng.choice([3, 5])))
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    threshold = rng.uniform(100, 190)
    img = img.point(lambda p: 255 if p > threshold else 0)
    if rng.random() < 0.5:  # scanned then resized: grey edges on a bilevel image
        f = rng.uniform(0.8, 1.25)
        img = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.LANCZOS)
    if rng.random() < 0.4:
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randint(1, 8)):
            x, y, r = rng.uniform(0, img.width), rng.uniform(0, img.height), rng.uniform(0.6, 1.8)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
    if rng.random() < pepper:
        draw = ImageDraw.Draw(img)
        for _ in range(round(img.width * img.height / 1000 * rng.uniform(0.3, 2.0))):
            x, y, size = rng.randrange(img.width), rng.randrange(img.height), rng.choice([1, 1, 2])
            draw.rectangle((x, y, x + size - 1, y + size - 1), fill=0)
    return img


def ink_share(img):
    return (np.asarray(img) < 128).mean()


def degrade(img, rng, scan=False):
    """img: 8-bit grayscale, white background, at RENDER_DPI. scan: always a binarized scan look."""
    scale = rng.choice([0.5, 0.5, rng.uniform(0.18, 0.5)])  # 300 dpi or lower-res
    if scan or rng.random() < 0.3:
        return scan_look(img, rng, scale, pepper=0.6 if scan else 0.25)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    if rng.random() < 0.25:  # keep a share of clean renders
        return img
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-0.8, 0.8), resample=Image.BICUBIC, expand=True, fillcolor=255)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    a = np.asarray(img, dtype=np.float32) / 255.0
    ink, paper = rng.uniform(0, 90), rng.uniform(180, 255)
    a = ink + a * (paper - ink)
    if rng.random() < 0.5:
        a += np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 14), a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(25, 85))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def render(job):
    idx, text, (font_name, (persian_digits, arabic_indic)), out_dir, seed, mode, fiction, diacritics = job
    rng = random.Random(seed)
    shown = text if rng.random() < diacritics else VOWEL_MARKS.sub("", text)
    if rng.random() < 0.08:
        shown = questions(shown, rng)
    if rng.random() < 0.5:
        shown = DIGIT_RANGE.sub(TATWEEL, shown)
    shown = add_kashida(shown, rng) if rng.random() < 0.25 else shown
    if fiction:
        if rng.random() < 0.25:
            shown = fiction_punctuation(shown, rng)
        if rng.random() < 0.08:
            shown = spaced_punctuation(shown)
        if arabic_indic and rng.random() < 0.3:
            shown = shown.translate(ARABIC_INDIC)
    dialogue = rng.random() < (0.10 if fiction else 0.03)
    if dialogue:
        shown = rng.choice(["ـ ", "ـ", "- ", "ـ... ", "ـ ... "] if fiction else ["ـ ", "ـ", "- "]) + shown
    if rng.random() < 0.4:
        shown = vary_spaces(shown, rng)
    label = canonical(shown)
    if dialogue:
        label = DIALOGUE_DASH.sub("- ", label)
    if persian_digits:
        label = label.translate(TO_PERSIAN_DIGITS)
    # text2image rewrites fonts.conf in its fontconfig dir on every call; a shared
    # dir lets parallel workers read it half-written and miss our fonts.
    fc_dir = ROOT / "work/fontconfig" / str(os.getpid())
    fc_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "work/tmp") as tmp:
        if mode == "t2i":
            result = text2image(shown, font_name, tmp, fc_dir, rng.randint(9, 16), 300,
                                ["--degrade_image=true", "--rotate_image=true",
                                 f"--exposure={rng.choice([-1, 0, 0, 1])}"])
        else:
            result = text2image(shown, font_name, tmp, fc_dir, rng.randint(9, 16), RENDER_DPI,
                                ["--degrade_image=false", "--rotate_image=false"])
    if result is None:
        return "render-failed"
    page, rendered = result
    if rendered != collections.Counter(c for c in shown if c not in IGNORED_IN_BOXES):
        return "text-mismatch"
    bbox = Image.eval(page.point(lambda p: 255 if p > 160 else 0), lambda p: 255 - p).getbbox()
    if bbox is None:
        return "empty"
    x0, y0, x1, y1 = bbox
    pad = [rng.randint(4, 20) * (2 if mode != "t2i" else 1) for _ in range(4)]
    line = page.crop((x0 - pad[0], y0 - pad[1], x1 + pad[2], y1 + pad[3]))
    name = f"{idx:07d}_{font_name.replace(' ', '')}"
    if mode != "t2i":
        clean, line = line, degrade(line, rng, scan=mode == "scan")
        if ink_share(line) < 0.35 * ink_share(clean):  # degraded beyond recognition
            return "faded"
    line.save(out_dir / f"{name}.png")
    (out_dir / f"{name}.gt.txt").write_text(label + "\n", encoding="utf-8")
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--fonts", choices=FONTS, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=0, help="corpus lines to skip (keeps eval text unseen)")
    ap.add_argument("--count", type=int, required=True, help="corpus lines to use")
    ap.add_argument("--latin-copies", type=int, default=1,
                    help="render lines containing Latin letters in this many different fonts")
    ap.add_argument("--degrade", choices=["mix", "scan", "t2i"], default="mix")
    ap.add_argument("--fiction", action="store_true", help="fiction-style punctuation and old typesetting")
    ap.add_argument("--diacritics", type=float, default=1.0, help="share of lines that keep their vowel marks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    opts = ap.parse_args()
    rng = random.Random(opts.seed)

    font_dir = ROOT / "work/fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "work/tmp").mkdir(parents=True, exist_ok=True)
    fonts = []
    for path in (p for group in opts.fonts for p in FONTS[group]):
        link = font_dir / path.name
        if not link.exists():
            link.symlink_to(path)
        fonts.append(font_info(path))
    shutil.rmtree(ROOT / "work/fontconfig", ignore_errors=True)  # font set may have changed

    lines = pathlib.Path(opts.corpus).read_text(encoding="utf-8").splitlines()[opts.skip:]
    out_dir = pathlib.Path(opts.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, used, uncovered = [], 0, 0
    for idx, text in enumerate(lines, start=opts.skip):
        if used == opts.count:
            break
        needed = {ord(c) for c in text if c not in " ‌"}
        usable = [(name, digits) for name, covered, digits in fonts if needed <= covered]
        if not usable:
            uncovered += 1
            continue
        used += 1
        # Latin letters are new output classes for the fas model and need extra examples.
        copies = opts.latin_copies if any(c.isascii() and c.isalpha() for c in text) else 1
        for font in rng.sample(usable, min(copies, len(usable))):
            jobs.append((idx, text, font, out_dir, rng.getrandbits(64), opts.degrade, opts.fiction, opts.diacritics))

    with Pool(opts.jobs) as pool:
        stats = collections.Counter(pool.imap_unordered(render, jobs, chunksize=32))
    print(f"{' '.join(opts.fonts)}: {dict(stats)}; {uncovered} lines skipped (no font covers them)")
    per_font = collections.Counter(j[2][0] for j in jobs)
    print("lines per font:", ", ".join(f"{k} {v}" for k, v in sorted(per_font.items())))


if __name__ == "__main__":
    main()
