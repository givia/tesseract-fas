#!/usr/bin/env python3
"""Cut real book lines for hand correction, and serve a local correction page.

    scripts/correct_lines.py cut work/books/machegooneh --test 12 --train 24
    scripts/correct_lines.py serve work/books/machegooneh      # http://127.0.0.1:8765
    scripts/correct_lines.py cut work/books/mosahebe --draft v3r40 --compare v2 --vocab BOOK.epub

`cut` picks text pages at random (skipping front matter, maps and tables: pages
with few lines or low word confidence, plus --exclude), splits them by page
into test and train, and crops every line the draft model found into
DIR/lines/SPLIT/p-NNN_LL.png. The draft reading, and the words where the draft
model is unsure or disagrees with a second model, go into DIR/lines/manifest.json.
With --vocab (the book's text as EPUB or plain text, e.g. an existing
transcription, however normalized) a word the book never uses is flagged too,
with the closest word of the book as a one-click suggestion.

`serve` shows each crop above its draft text. A confirmed line is written as
DIR/lines/SPLIT/NAME.gt.txt (canonical() orthography), ready for prep_lstmf.py
and evaluate.py; all edits are kept in DIR/lines/corrections.json. The server
listens on 127.0.0.1 only: the scans never leave this machine.
"""
import argparse
import html
import http.server
import json
import pathlib
import random
import re
import statistics
import zipfile

from PIL import Image
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from fa_text import canonical, relaxed

LINE = re.compile(r"<span class='(ocr_line|ocr_header|ocr_caption|ocr_textfloat)'[^>]*title=\"bbox (\d+) (\d+) (\d+) (\d+);")
WORD = re.compile(r"<span class='ocrx_word'[^>]*x_wconf (\d+)'[^>]*>(.*?)</span>", re.S)
# Running header: "96 ▣ ما چگونه، ما شدیم: ..." or "فصل پنجم: ... ▣ 247"; the box glyph reads as "B".
HEADER = re.compile(r"(^|\s)B(\s|$)")


def hocr_lines(path):
    """[(bbox, [(word, confidence)])] in reading order."""
    text = path.read_text(encoding="utf-8")
    starts = list(LINE.finditer(text))
    lines = []
    for i, m in enumerate(starts):
        chunk = text[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        words = [(html.unescape(re.sub("<[^>]+>", "", w)), int(c)) for c, w in WORD.findall(chunk)]
        lines.append((tuple(int(g) for g in m.groups()[1:]), [(w, c) for w, c in words if w.strip()]))
    return lines


def overlap(a, b):
    v = max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, max(a[3], b[3]) - min(a[1], b[1]))
    h = max(0, min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2] - a[0], b[2] - b[0]))
    return v * h


def fold(word):
    return re.sub(r"\s", "", relaxed(word))  # relaxed() turns ZWNJ into a space


def vocab_key(word):
    return re.sub(r"\W", "", relaxed(word))  # no ZWNJ, spaces, punctuation, vowel marks or ezafe hamza


class Vocabulary:
    """Words of a reference text, for spotting OCR words the book never uses."""

    def __init__(self, path):
        path = pathlib.Path(path)
        if path.suffix == ".epub":
            with zipfile.ZipFile(path) as z:
                text = " ".join(html.unescape(re.sub("<[^>]+>", " ", z.read(n).decode("utf-8")))
                                for n in sorted(z.namelist()) if n.endswith((".xhtml", ".html")))
        else:
            text = path.read_text(encoding="utf-8")
        counts = {}
        for word in canonical(text).split():
            for form in (word, *word.split("\u200c")):
                key = vocab_key(form)
                if key:
                    counts.setdefault(key, {}).setdefault(form.strip(".,،؛:؟!«»()[]\"'"), 0)
                    counts[key][form.strip(".,،؛:؟!«»()[]\"'")] += 1
        self.display = {k: max(forms, key=forms.get) for k, forms in counts.items()}
        self.keys = list(self.display)

    def known(self, word):
        parts = [vocab_key(p) for p in re.split("[\u200c ]", canonical(word))]
        return vocab_key(word) in self.display or all(p in self.display for p in parts if p)

    def closest(self, key):
        """(display form, edit distance) of the closest reference word, or None."""
        if key in self.display:
            return self.display[key], 0
        found = process.extract(key, self.keys, scorer=Levenshtein.distance, score_cutoff=max(1, len(key) // 3), limit=20)
        if not found:
            return None
        # Among equally close words prefer the same length: OCR mostly misreads letters
        # (سوجود = موجود) rather than inventing or dropping them.
        best = min(found, key=lambda f: (f[1], abs(len(f[0]) - len(key))))
        return self.display[best[0]], best[1]

    def suggest(self, word):
        """Closest reading in the reference text's words, keeping the OCR word's punctuation.

        Also tries two words run together ("باشدکه" -> "باشد که", "سی‌گویندکه" -> "می‌گویند که"):
        a single-word suggestion would silently drop one of them. None if nothing is close.
        """
        m = re.fullmatch(r"(\W*)(.*?)(\W*)", canonical(word))
        key = vocab_key(m[2])
        if len(key) < 2:
            return None
        options = []
        whole = self.closest(key)
        if whole:
            options.append((whole[1] + 0.5, whole[0]))  # a split costs a space; prefer it only when better
        # Where the OCR put a ZWNJ ("سی‌گوید") the word boundary is usually right: split elsewhere.
        zwnj_at = {len(vocab_key(m[2][:j])) for j, ch in enumerate(m[2]) if ch == "\u200c"}
        for i in range(1, len(key)):
            left, right = key[:i], key[i:]
            if i in zwnj_at or (min(len(left), len(right)) < 2 and "و" not in (left, right)):
                continue
            if left in self.display or right in self.display:
                l, r = self.closest(left), self.closest(right)
                if l and r:
                    options.append((l[1] + r[1] + 1, f"{l[0]} {r[0]}"))
        if not options:
            return None
        return m[1] + min(options)[1] + m[3]


def cut(opts):
    root = pathlib.Path(opts.book_dir)
    draft_dir, other_dir = root / "ocr" / opts.draft, root / "ocr" / opts.compare
    exclude = {int(p) for part in opts.exclude.split(",") if part
               for p in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))}
    pages = []
    for hocr in sorted(draft_dir.glob("*.hocr")):
        lines = hocr_lines(hocr)
        confs = [c for _, words in lines for _, c in words]
        if int(hocr.stem[2:]) in exclude or len(lines) < 15 or not confs or statistics.mean(confs) < 85:
            continue
        gray = Image.open(root / f"{hocr.stem}.png").mode != "1"
        pages.append((hocr.stem, gray))
    rng = random.Random(opts.seed)
    rng.shuffle(pages)
    # Keep the share of grayscale scans the book has, in both splits.
    gray_share = sum(g for _, g in pages) / len(pages)
    chosen, taken = {}, set()
    for split, n in (("test", opts.test), ("train", opts.train)):
        n_gray = round(n * gray_share) or (1 if gray_share else 0)
        for want_gray, count in ((True, n_gray), (False, n - n_gray)):
            picks = [p for p, g in pages if g == want_gray and p not in taken][:count]
            taken.update(picks)
            chosen.update({p: split for p in picks})

    vocab = Vocabulary(opts.vocab) if opts.vocab else None
    out = root / "lines"
    manifest = []
    for page, split in sorted(chosen.items(), key=lambda kv: (kv[1] != "test", kv[0])):
        (out / split).mkdir(parents=True, exist_ok=True)
        image = Image.open(root / f"{page}.png")
        other = [(box, " ".join(w for w, _ in words)) for box, words in hocr_lines(other_dir / f"{page}.hocr")]
        for i, (box, words) in enumerate(hocr_lines(draft_dir / f"{page}.hocr")):
            draft = canonical(" ".join(w for w, _ in words))
            if len(draft) < 4 or HEADER.search(draft):
                continue
            x0, y0, x1, y1 = box
            name = f"{page}_{i:02d}"
            image.crop((max(0, x0 - 10), max(0, y0 - 5), min(image.width, x1 + 10),
                        min(image.height, y1 + 5))).save(out / split / f"{name}.png")
            best = max(other, key=lambda o: overlap(box, o[0]), default=None)
            other_words = {fold(w) for w in best[1].split()} if best and overlap(box, best[0]) > 0.5 else set()
            # The second model is often the one that is wrong, so its disagreement only counts
            # where the draft model is not confident either.
            check = []
            for w, c in words:
                unknown = vocab is not None and not vocab.known(w)
                if unknown or c < opts.min_conf or (c < opts.agree_conf and fold(w) not in other_words):
                    check.append([canonical(w), vocab.suggest(w) if unknown else None])
            manifest.append({"name": name, "split": split, "page": page, "draft": draft, "check": check})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=0), encoding="utf-8")
    for split in ("test", "train"):
        ps = sorted(p for p, s in chosen.items() if s == split)
        print(f"{split}: {len(ps)} pages, {sum(m['split'] == split for m in manifest)} lines: "
              + " ".join(p[2:] for p in ps))


PAGE = """<!doctype html>
<html lang="fa"><head><meta charset="utf-8"><title>Line Corrections</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#fafaf8; --fg:#1b1b1b; --muted:#6b6b6b; --line:#ddd; --ok:#e7f4e8; --okb:#3a8a45;
        --skip:#f1f1f1; --chip:#fff2c2; --card:#fff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#1a1a1a; --fg:#e8e8e8; --muted:#a0a0a0; --line:#333; --ok:#1f3323; --okb:#6cc27a;
  --skip:#262626; --chip:#4a4020; --card:#222; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; z-index:1; }
header b { font-variant-numeric:tabular-nums; }
main { max-width:1100px; margin:0 auto; padding:8px 16px 40vh; }
details { color:var(--muted); margin:6px 0; }
h2 { font-size:17px; margin:28px 0 6px; }
.item { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px; margin:8px 0; }
.item.ok { background:var(--ok); border-color:var(--okb); }
.item.skip { background:var(--skip); opacity:.55; }
.item img { display:block; max-width:100%; margin-inline-start:auto; background:#fff; }
.row { display:flex; gap:6px; align-items:center; margin-top:6px; }
input { flex:1; min-width:0; font:22px/1.6 "Vazirmatn", "Noto Naskh Arabic", system-ui, sans-serif; padding:2px 8px;
        border:1px solid var(--line); border-radius:4px; background:var(--card); color:var(--fg); }
button { font:inherit; padding:4px 10px; border-radius:4px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
.meta { font-size:12px; color:var(--muted); display:flex; gap:6px; flex-wrap:wrap; margin-top:4px; direction:rtl; }
.chip { background:var(--chip); border-radius:3px; padding:0 5px; font-size:15px; color:var(--fg); border:0; font-family:inherit; }
button.chip { cursor:pointer; text-decoration:underline dotted; }
</style></head><body>
<header>Checked <b id="done">0</b> / <b id="total">0</b> lines (test <b id="dtest">0</b>/<b id="ttest">0</b>,
train <b id="dtrain">0</b>/<b id="ttrain">0</b>) · skipped <b id="skipped">0</b> ·
<button id="next">Jump to next unchecked</button>
<details><summary>How to correct</summary>
Type exactly what is printed. <b>Enter</b> saves the line as correct and moves on; <b>Ctrl+Enter</b> skips a
line that is not one clean text line (a cut-off crop, two lines, a map label). Yellow words are where the model
is unsure or two models disagree, so look there first; "word → suggestion" chips come from the book's
text, and clicking one puts the suggestion into the line (that text drops the ezafe hamza ۀ and changes
spacing, so check the image before accepting).
Conventions: ZWNJ (نیم‌فاصله) where the print joins words without a space (می‌شود), a space where it
leaves a word gap; digits as printed (Persian ۱۲۳ or Latin 123); a dash or a standalone tatweel as "-";
no kashida (stretched letters are typed normally); « » ، ؛ ؟ as printed; ignore bold/italic; Latin text
as printed. Saving normalizes Arabic ي/ك to Persian ی/ک. Test pages come first: they matter most.
</details></header>
<main id="list"></main>
<script>
const items = __ITEMS__;
const state = __STATE__;
const list = document.getElementById("list");
let lastPage = null;
for (const it of items) {
  if (it.page !== lastPage) {
    const h = document.createElement("h2");
    h.textContent = `${it.split === "test" ? "Test" : "Train"} · page ${+it.page.slice(2)}`;
    list.append(h); lastPage = it.page;
  }
  const s = state[it.name] || {};
  const div = document.createElement("div");
  div.className = "item" + (s.status ? " " + s.status : "");
  div.id = it.name;
  div.innerHTML = `<img loading="lazy" alt="line ${it.name}" src="img/${it.split}/${it.name}.png">
    <div class="row"><button class="skipb" title="Ctrl+Enter">skip</button>
    <button class="okb" title="Enter">✓</button><input dir="rtl" lang="fa" spellcheck="false"></div>
    <div class="meta"></div>`;
  const input = div.querySelector("input");
  input.value = s.text ?? it.draft;
  const meta = div.querySelector(".meta");
  for (const entry of it.check) {
    const [w, s] = Array.isArray(entry) ? entry : [entry, null];
    const c = document.createElement(s ? "button" : "span");
    c.className = "chip";
    c.textContent = s ? `${w} → ${s}` : w;
    if (s) c.onclick = () => { input.value = input.value.replace(w, s); input.focus(); };
    meta.append(c);
  }
  const save = async (status) => {
    const r = await fetch("save", {method: "POST", body: JSON.stringify({name: it.name, text: input.value, status})});
    const saved = await r.json();
    input.value = saved.text;
    state[it.name] = saved;
    div.className = "item " + status;
    counts();
  };
  div.querySelector(".okb").onclick = () => save("ok").then(next);
  div.querySelector(".skipb").onclick = () => save("skip").then(next);
  input.addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    save(e.ctrlKey || e.metaKey ? "skip" : "ok").then(next);
  });
  list.append(div);
}
function next() {
  const it = items.find(i => !(state[i.name] || {}).status);
  if (!it) return;
  const el = document.getElementById(it.name);
  el.scrollIntoView({block: "center"});
  el.querySelector("input").focus({preventScroll: true});
}
function counts() {
  const c = {done: 0, skipped: 0, test: 0, train: 0, ttest: 0, ttrain: 0};
  for (const it of items) {
    const st = (state[it.name] || {}).status;
    c["t" + it.split]++;
    if (st) { c.done++; c[it.split]++; }
    if (st === "skip") c.skipped++;
  }
  for (const [k, v] of Object.entries({done: c.done, total: items.length, dtest: c.test, ttest: c.ttest,
      dtrain: c.train, ttrain: c.ttrain, skipped: c.skipped})) document.getElementById(k).textContent = v;
}
document.getElementById("next").onclick = next;
counts(); next();
</script></body></html>"""


def serve(opts):
    out = pathlib.Path(opts.book_dir) / "lines"
    items = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    by_name = {it["name"]: it for it in items}
    state_file = out / "corrections.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                page = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)) \
                           .replace("__STATE__", json.dumps(state, ensure_ascii=False))
                return self.send(page.encode(), "text/html; charset=utf-8")
            m = re.fullmatch(r"/img/(test|train)/(p-\d+_\d+)\.png", self.path)
            if m and m[2] in by_name:
                return self.send((out / m[1] / f"{m[2]}.png").read_bytes(), "image/png")
            self.send_error(404)

        def do_POST(self):
            if self.path != "/save":
                return self.send_error(404)
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            it = by_name[req["name"]]
            text, status = canonical(req["text"]).replace("\n", " "), req["status"]
            state[it["name"]] = {"text": text, "status": status}
            gt = out / it["split"] / f"{it['name']}.gt.txt"
            if status == "ok" and text:
                gt.write_text(text + "\n", encoding="utf-8")
            else:
                gt.unlink(missing_ok=True)
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=0), encoding="utf-8")
            self.send(json.dumps(state[it["name"]], ensure_ascii=False).encode(), "application/json")

        def log_message(self, *args):
            pass

    print(f"serving {len(items)} lines on http://127.0.0.1:{opts.port}/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", opts.port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("book_dir")
    c.add_argument("--draft", default="v2", help="OCR label whose reading pre-fills the lines")
    c.add_argument("--compare", default="stock", help="second OCR label, for flagging disagreements")
    c.add_argument("--test", type=int, default=12, help="test pages")
    c.add_argument("--train", type=int, default=24, help="train pages")
    c.add_argument("--exclude", default="", help="page ranges to leave out, e.g. 1-11,93-97")
    c.add_argument("--min-conf", type=int, default=60, help="flag draft words below this confidence")
    c.add_argument("--agree-conf", type=int, default=85,
                   help="also flag words below this confidence that the second model read differently")
    c.add_argument("--vocab", help="the book's text (EPUB or plain text) for flagging words the book never uses")
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("serve")
    s.add_argument("book_dir")
    s.add_argument("--port", type=int, default=8765)
    opts = ap.parse_args()
    cut(opts) if opts.cmd == "cut" else serve(opts)


if __name__ == "__main__":
    main()
