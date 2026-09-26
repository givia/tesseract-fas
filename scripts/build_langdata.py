#!/usr/bin/env python3
"""Build the dictionary files combine_lang_model uses (wordlist, punc, numbers).

The upstream langdata_lstm/fas wordlist mixes Arabic yeh/kaf, diacritics and bidi
controls, so it is rebuilt from Persian Wikipedia word frequencies. The punc and
numbers patterns are kept, cleaned of bidi controls, and extended with Persian
marks and common number shapes.

In these pattern files a space is a placeholder, not a space: in punc it stands
for any word, in numbers for any digit (Dict::char_for_dawg maps every digit to
Dawg::kPatternUnicharID, unichar id 0, which is the space). So they must not go
through canonical(), which would strip and collapse those spaces.
"""
import argparse
import collections
import pathlib
import re

import pyarrow.parquet as pq

from build_corpus import PERSIAN
from fa_text import BIDI_CONTROLS, DIACRITICS, canonical

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORD = re.compile(f"[{PERSIAN}‌]+")

# " " = one digit (see module docstring).
EXTRA_NUMBERS = ["    /  /  ", "  /  /    ", "    /  ", " ٫ ", "  ٫  ", " ٪", "  ٪", "   ٪", " ٬   ", "  ٬   ",
                 "   ٬   ", " ٬   ٬   ", "    -    ", "   -    ", "(   )"]
STRIP_BIDI = str.maketrans({c: None for c in BIDI_CONTROLS})


def upstream_patterns(name):
    lines = (ROOT / "external/langdata/fas" / name).read_text(encoding="utf-8").splitlines()
    return {l.translate(STRIP_BIDI) for l in lines if l.translate(STRIP_BIDI)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--out", default="work/langdata", help="writes fas.wordlist/.punc/.numbers here")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=200_000)
    opts = ap.parse_args()
    out = pathlib.Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    freq = collections.Counter()
    for path in opts.parquet:
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            freq.update(w.strip("‌") for w in WORD.findall(DIACRITICS.sub("", canonical(text))))
    words = [w for w, n in freq.most_common(opts.max_words) if n >= opts.min_count and w]
    (out / "fas.wordlist").write_text("\n".join(words) + "\n", encoding="utf-8")

    punc = upstream_patterns("fas.punc")
    punc |= {p.translate(str.maketrans(",;?", "،؛؟")) for p in punc}
    (out / "fas.punc").write_text("\n".join(sorted(punc)) + "\n", encoding="utf-8")

    numbers = sorted(upstream_patterns("fas.numbers") | set(EXTRA_NUMBERS))
    (out / "fas.numbers").write_text("\n".join(numbers) + "\n", encoding="utf-8")
    print(f"{len(words)} words (of {len(freq)} distinct), {len(punc)} punc patterns, {len(numbers)} number patterns")
    print("top words:", " ".join(words[:25]))


if __name__ == "__main__":
    main()
