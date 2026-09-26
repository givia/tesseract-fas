#!/usr/bin/env python3
"""Build training text lines from Persian Wikipedia.

Articles are split into sentences, normalized with canonical(), chopped into
lines of realistic printed length, filtered to the target character set, and
de-templated (Persian Wikipedia has ~100k near-identical village stubs).
Lines containing characters the stock model lacks (Persian comma, Latin, ...)
are oversampled so the new output classes get enough examples.
"""
import argparse
import collections
import pathlib
import random
import re

import pyarrow.parquet as pq

from fa_text import canonical

PERSIAN = "اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأؤإئةۀ"
DIACRITICS = "ًٌٍَُِّْ"
DIGITS = "۰۱۲۳۴۵۶۷۸۹0123456789"
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
PUNCT = ".،؛؟!:«»()[]-–/\"'٪%+=*,;?@&#…"
ALLOWED = set(PERSIAN + DIACRITICS + DIGITS + LATIN + PUNCT + " ‌")

# Characters the stock fas model cannot emit; lines containing them are boosted.
NEW_PUNCT = set("،ؤإۀ٪%?;=@&'–…")
TO_LATIN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

SENTENCE_END = re.compile(r"(?<=[.؟!])\s+")


def documents(path):
    if str(path).endswith(".parquet"):
        yield from pq.read_table(path, columns=["text"]).column("text").to_pylist()
    else:  # plain text, e.g. langdata_lstm's fas.training_text
        yield from open(path, encoding="utf-8")


def sentences(paths):
    for path in paths:
        for text in documents(path):
            for para in text.split("\n"):
                para = canonical(para)
                if len(para) >= 20 and not para.startswith(("|", "{", "*")):
                    yield from SENTENCE_END.split(para)


def chop(sentence, rng, lo=12, hi=64):
    """Cut a sentence into lines of lo..hi chars at word boundaries."""
    words, line = sentence.split(), []
    target = rng.randint(lo, hi)
    for w in words:
        if line and len(" ".join(line + [w])) > target:
            yield " ".join(line)
            line, target = [], rng.randint(lo, hi)
        line.append(w)
    if line and len(" ".join(line)) >= lo:
        yield " ".join(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="Wikipedia .parquet files or plain .txt files")
    ap.add_argument("--out", default="work/corpus/lines.txt")
    ap.add_argument("--lines", type=int, default=200_000)
    ap.add_argument("--latin-share", type=float, default=0.12, help="share of lines with Latin letters")
    ap.add_argument("--punct-share", type=float, default=0.13, help="share of lines with other new chars")
    ap.add_argument("--latin-digits", type=float, default=0.35,
                    help="share of digit-bearing lines rewritten with Latin digits (both occur in print)")
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()
    rng = random.Random(opts.seed)

    seen, per_template = set(), collections.Counter()
    pools = {"latin": [], "punct": [], "plain": []}
    for sent in sentences(opts.inputs):
        for line in chop(sent, rng):
            if not set(line) <= ALLOWED or sum(c in PERSIAN for c in line) < len(line) / 3:
                continue
            # tesstrain's bidi step takes the paragraph direction from the first
            # strong character; a Latin-first line would get a mirrored label.
            if next(c for c in line if c in PERSIAN or c in LATIN) in LATIN:
                continue
            key = re.sub(r"\d|[۰-۹]", "#", line)
            template = " ".join(key.split()[:3])
            if key in seen or per_template[template] >= 20:
                continue
            seen.add(key)
            per_template[template] += 1
            chars = set(line)
            pools["latin" if chars & set(LATIN) else "punct" if chars & NEW_PUNCT else "plain"].append(line)

    lines = []
    for name, share in [("latin", opts.latin_share), ("punct", opts.punct_share)]:
        lines += rng.sample(pools[name], min(len(pools[name]), int(opts.lines * share)))
    lines += rng.sample(pools["plain"], min(len(pools["plain"]), opts.lines - len(lines)))
    lines = [l.translate(TO_LATIN_DIGITS) if rng.random() < opts.latin_digits else l for l in lines]
    rng.shuffle(lines)
    out = pathlib.Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    freq = collections.Counter("".join(lines))
    print(f"{len(lines)} lines; available per pool:", {k: len(v) for k, v in pools.items()})
    print("rarest target chars:", ", ".join(f"{c!r}:{freq[c]}" for c in sorted(ALLOWED, key=freq.__getitem__)[:25]))


if __name__ == "__main__":
    main()
