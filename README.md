# Better Persian (fas) OCR for Tesseract

Fine-tuning Tesseract's `fas` LSTM model, with a reproducible pipeline and an
honest evaluation. The result, `fas_print`, is submitted to
[tesseract-ocr/tessdata_contrib](https://github.com/tesseract-ocr/tessdata_contrib)
as [PR #19](https://github.com/tesseract-ocr/tessdata_contrib/pull/19)
(model card in `dist/tessdata_contrib/fas_print/`); the bugs found on the way
are filed upstream, see `docs/upstream/README.md` for the list
(tesseract#4627/#4628 SIGFPE, tesseract#4629 reading direction,
tesstrain#430/#431, langdata_lstm#66).

## Why the stock model needs work

The official `fas` model (`tessdata_best`, `synth20170629`) has systematic
problems that no amount of preprocessing fixes:

| Problem | Cause (found in `langdata_lstm/fas` and the model's unicharset) |
|---|---|
| Can never output the Persian comma `،` | `forbidden_characters` bans U+0600–U+0615, which contains U+060C, while `desired_characters` asks for it |
| English words become garbage | The 95-character unicharset has no Latin letters at all |
| Mixes `ي`/`ی` and `ك`/`ک` in output | ~32% of yeh and ~30% of kaf in `fas.training_text` are the Arabic forms |
| Emits stray LRM/RLM, Arabic-Indic `٩` | Bidi control characters and `٩` are in the training text/unicharset |
| Weak on the fonts Iranian documents use | `okfonts.txt` lists only DejaVu, FreeMono, FreeSerif and Tahoma |
| Also missing: `ٔ ؤ إ % ٪ ? ; = @` | Absent from training text / unicharset |

Related upstream reports: tessdata_best#11 (Arabic comma, 2017),
tessdata#70 (Persian feedback), tesseract#4582 (lines dropped by the textord
noise filter in dotted Arabic-script text).

## What this pipeline does differently

- **Training text**: real sentences from Persian Wikipedia instead of word
  salad, normalized to one orthography (`scripts/fa_text.py`: Persian yeh/kaf,
  no bidi controls, no tatweel, NFC), with lines containing Latin words and
  Persian punctuation oversampled, and a share of numbers in Latin digits (both
  occur in print).
- **Fonts**: 43 faces: Naskh book faces (Nazli, Amiri, Scheherazade New,
  Mirza, Noto Naskh), modern Persian sans faces (Vazirmatn, Sahel, Samim,
  Shabnam, Tanha, Gandom, Nahid), the stock DejaVu/Noto Sans, and the 12
  classic Iranian "B" faces (B Nazanin, B Lotus, B Mitra, B Zar, …) that most
  Iranian print uses. Each line is rendered only in a font that covers all its
  characters, so nothing is silently dropped or substituted.
- **Degradation**: resolution loss down to ~110 dpi, blur, noise, contrast,
  JPEG artifacts, slight skew, binarized-scan look with stroke-weight changes
  and pepper noise; simulated kashida justification, narrow and wide word
  spaces, thinned diacritics (in the image only, not the label).
- **Real lines**: `scripts/correct_lines.py` cuts text lines out of scanned
  books using the OCR's own line boxes, flags doubtful words (low confidence,
  disagreement with a second model, unknown to a vocabulary) and serves a
  local correction page; corrected lines go into training repeated 40×.
- **Rehearsal**: a share of Google's original `fas` training text rendered
  with `text2image`-style degradation, so the model keeps the stock model's
  robustness on poor scans (the first fine-tune without it regressed on real
  pages).
- **Correct RTL labels**: tesstrain's WordStr box writer deletes ZWNJ and
  mirrors brackets (`docs/upstream/`); `scripts/wordstr_box.py` does the
  visual ordering without those two steps.
- **Dictionary**: wordlist rebuilt from Wikipedia frequencies; punc/number
  patterns cleaned of bidi controls and extended with `، ؛ ٪ ٫ ٬` shapes.

## Evaluation

None of the test sets is used for training. CER strict counts orthography
(e.g. `،` vs `,`); CER relaxed folds digit and punctuation script, ZWNJ vs
space and diacritics, to be fair to third-party transcriptions. Page CER on
multi-column layouts is dominated by reading order, so word recall
(order-free) is the more meaningful number there.

| Test set | stock `fas` | `fas_v2` | `fas_v4` = `fas_print` |
|---|---|---|---|
| 190 real lines, 1980s phototypeset book (`work/lines/mosahebe_test`), CER strict | 5.38% | 3.78% | **1.14%** |
| 321 real lines, 2010s book scan (`work/lines/mcms_test`), CER strict | 4.65% | 1.98% | **1.25%** |
| 36 random real lines, two older novels (`work/lines/realbook_rand`), CER strict | 6.51% | 3.52% | **2.59%** |
| Iranian constitution, 4 pages, `--psm 6` (`work/lines/const_pages`), CER | 1.30% | 0.34% | **0.25%** |
| Held-out fonts, 2,000 synthetic lines (`work/lines/holdout`), CER | 5.11% | **1.72%** | 1.79% |
| 74 real printed benchmark pages, word recall | 50.66%¹ | 51.30% | **53.04%** |

¹ with `-c textord_noise_rejrows=0`, which the fine-tuned models carry in
their own config.

The integer (fast) conversion of v4 is within 0.05 points of the float model
on every real-scan set and 0.9 points worse on the synthetic holdout.

### Rounds

| Model | Data | Outcome |
|---|---|---|
| v1 | 150k Wikipedia lines, 31 open fonts, 160k iterations | much better on synthetic lines, worse than stock on real pages |
| v2 | + B fonts, fiction typesetting, rehearsal of Google's text; 100k iterations | first model better than stock everywhere |
| v3 | re-rendered with scan-look degradation, kashida, spacing and question drills | better on modern print, but a `م`→`س` habit on bold phototype |
| v3r40 | v3 + 665 corrected real lines ×40 | best on the 2010s book (1.15%) |
| v4 | v3 + 665 + 617 real lines from two books, ×40 each | fixes the bold-face problem (3.94% → 0.63% relaxed CER); release candidate |

## Reproduce

```sh
scripts/fetch.sh                                     # models, langdata, fonts, corpus, benchmark
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_corpus.py external/corpus/fawiki-1.parquet
.venv/bin/python scripts/build_langdata.py external/corpus/fawiki-1.parquet

# Synthetic line sets (the seeds below are the ones v4 was trained with)
R=.venv/bin/python\ scripts/render_lines.py
$R work/corpus/lines.txt --fonts train --out work/lines/v3/wiki --count 120000 --seed 41
$R work/corpus/lines.txt --fonts book --out work/lines/v3/book --count 100000 --seed 42
$R work/corpus/lines.txt --fonts train book --fiction --out work/lines/v3/fiction --skip 150000 --count 40000 --seed 43
$R work/corpus/google_lines.txt --fonts train book --degrade t2i --diacritics 0.2 --out work/lines/v3/rehearsal --count 40000 --seed 44
$R work/corpus/lines.txt --fonts book --degrade scan --out work/lines/v3/scan --count 30000 --skip 120000 --seed 45
# Evaluation sets (text never used for training)
$R work/corpus/lines.txt --fonts holdout --out work/lines/holdout --skip 194000 --count 2000 --seed 2

# Real lines from a scanned book: OCR the pages, cut lines, correct them in the browser
.venv/bin/python scripts/ocr_pages.py work/books/BOOK --model draft:work/data:fas_v3r40 --model v2:work/data:fas_v2
.venv/bin/python scripts/correct_lines.py cut work/books/BOOK --draft draft --compare v2 --test 6 --train 20
.venv/bin/python scripts/correct_lines.py serve work/books/BOOK      # http://127.0.0.1:8765
# Repeat the corrected lines 40× (symlinked copies) and assemble the training set
for c in $(seq -w 0 19); do mkdir -p work/lines/v3/real/c$c; ln -s $PWD/work/books/BOOK/lines/train/* work/lines/v3/real/c$c/; done
mkdir -p work/lines/train_v4 && for d in wiki book fiction rehearsal scan real; do ln -sfn ../v3/$d work/lines/train_v4/$d; done
ln -sfn ../v3/real work/lines/train_v4/real_again

scripts/train.sh fas_v4 work/lines/train_v4 MAX_ITERATIONS=100000
.venv/bin/python scripts/evaluate.py --set work/lines/holdout --model stock:external/tessdata_best:fas:"--psm 13" \
    --model v4:work/data:fas_v4:"--psm 13"
.venv/bin/python scripts/evaluate.py --set bench --page-type printed --model v4:work/data:fas_v4

# Package for tessdata_contrib (float and integer variants)
lstmtraining --stop_training --convert_to_int --continue_from work/data/fas_v4/checkpoints/fas_v4_checkpoint \
    --traineddata work/data/fas_v4/fas_v4.traineddata --model_output work/data/fas_v4_int.traineddata
```

Requires Tesseract 5 with training tools (`text2image`, `lstmtraining`,
`combine_tessdata`), GNU make ≥ 4.2, `bsdtar`, `fc-query`. Training runs on
CPU (about 45 minutes per 100k iterations on 32 cores; `lstmtraining` uses at
most 8 threads, so several runs fit side by side).

## Repository layout

- `scripts/` — the pipeline (see the docstring at the top of each script).
- `data/` — benchmark page labels, the model config embedded in the traineddata.
- `dist/tessdata_contrib/fas_print/` — the contribution: model card, `best/`, `fast/`.
- `docs/upstream/` — issue and PR drafts with evidence, and the patches.
- `repro/` — image reproductions for the upstream issues (rendered from
  Wikipedia text in OFL fonts, safe to attach publicly).
- `work/` (ignored) — rendered lines, book scans, models, evaluation caches.

## Licenses of inputs

Training text: Persian Wikipedia (CC BY-SA 4.0) and the `langdata_lstm/fas`
training text (Apache-2.0). Fonts are used only to render training images and
are not redistributed: Vazirmatn, Parastoo, Estedad and the Google Fonts faces
are OFL; FarsiWeb fonts are GPL-2; Sahel, Samim, Shabnam, Tanha, Gandom and
Nahid are by Saber Rastikerdar (see their repos); the "B" faces circulate
freely without a clear licence. The scanned books used for real training and
test lines are copyrighted; their line images stay local and only the model
is published. The benchmark is used for evaluation only and is not
redistributed.
