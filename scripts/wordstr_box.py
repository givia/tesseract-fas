#!/usr/bin/env python3
"""WordStr box files for RTL line images, in the label order Tesseract's LSTM uses.

Replaces tesstrain's generate_wordstr_box.py, whose bidi.get_display() call
corrupts Persian labels in two ways:
  * it drops ZWNJ (bidi class BN, removed by UBA rule X9), so a fine-tuned
    model unlearns the half-space;
  * it mirrors every bracket and guillemet at an RTL level (rule L4).
The label has to be whatever Tesseract turns back into the right logical text.
Tesseract reverses the characters of RTL (and neutral) words, so there a
mirrored glyph must be labelled with its logical code point: the stock fas
model reads the right-edge bracket of "(...)" in RTL text as U+0028. Words that
contain Latin letters and no RTL letters are output left to right unchanged, so
there the label is the drawn glyph, i.e. mirrored ("(multiple)" inside Persian
text). Otherwise the model learns to emit ")...(" and "»...«".
The label is the visual (left-to-right image) order from the UBA with an RTL
paragraph direction, as the images are rendered.

    scripts/wordstr_box.py -i line.png -t line.gt.txt > line.box
"""
import argparse
import unicodedata

from bidi import algorithm as uba
from PIL import Image

JOINERS = {"‌", "‍"}  # ZWNJ, ZWJ


def visual_order(text: str) -> str:
    storage = uba.get_empty_storage()
    storage["base_level"] = uba.PARAGRAPH_LEVELS["R"]
    storage["base_dir"] = "R"
    uba.get_embedding_levels(text, storage, False, False)
    chars = storage["chars"]
    for i, ch in enumerate(chars):
        if ch["ch"] in JOINERS:
            # Survive rule X9 by taking a neighbour's class, which leaves ordering unchanged.
            neighbour = chars[i - 1] if i else chars[i + 1] if i + 1 < len(chars) else None
            ch["type"] = ch["orig"] = neighbour["orig"] if neighbour else "ON"
    uba.explicit_embed_and_overrides(storage, False)
    uba.resolve_weak_types(storage, False)
    uba.resolve_neutral_types(storage, False)
    uba.resolve_implicit_levels(storage, False)
    uba.reorder_resolved_levels(storage, False)
    chars = storage["chars"]
    start = 0
    for end in [i for i, ch in enumerate(chars) if ch["ch"] == " "] + [len(chars)]:
        word = chars[start:end]
        classes = {ch["orig"] for ch in word}
        if "L" in classes and not classes & {"R", "AL"}:  # Tesseract outputs this word as is
            for ch in word:
                if ch["level"] % 2:
                    ch["ch"] = uba.MIRRORED.get(ch["ch"], ch["ch"])
        start = end + 1
    return "".join(ch["ch"] for ch in chars)


def wordstr_box(image_path, text) -> str:
    width, height = Image.open(image_path).size
    line = visual_order(unicodedata.normalize("NFC", text.strip()))
    return f"WordStr 0 0 {width} {height} 0 #{line}\n\t 0 0 {width} {height} 0\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-t", "--txt", required=True)
    opts = ap.parse_args()
    lines = open(opts.txt, encoding="utf-8").read().strip().split("\n")
    if len(lines) != 1:
        raise SystemExit(f"{opts.txt}: expected exactly one line, got {len(lines)}")
    print(wordstr_box(opts.image, lines[0]), end="")


if __name__ == "__main__":
    main()
