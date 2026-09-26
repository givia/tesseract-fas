# Upstream contributions

The texts below are what was filed on 2026-09-26. Evidence and reproduction
files live in this repo; numbers come from `scripts/evaluate.py`.

| # | Where | Type | Filed as |
|---|---|---|---|
| 1 | tesseract-ocr/tesstrain | PR: RTL/Indic WordStr labels drop ZWJ/ZWNJ and mirror brackets | [tesstrain#430](https://github.com/tesseract-ocr/tesstrain/pull/430), patch `tesstrain-wordstr-bidi.patch` |
| 2 | tesseract-ocr/tesstrain | PR: `$(ALL_LSTMF)` recipe fails when the output dir does not exist | [tesstrain#431](https://github.com/tesseract-ocr/tesstrain/pull/431) |
| 3 | tesseract-ocr/tesseract#4582 | Comment: image repro + measured impact, `textord_noise_rejrows` is the culprit | [comment](https://github.com/tesseract-ocr/tesseract/issues/4582#issuecomment-5849064524), image in `repro/` |
| 4 | tesseract-ocr/langdata_lstm | Issue: `fas` data problems (comma forbidden, Arabic yeh/kaf, bidi controls) | [langdata_lstm#66](https://github.com/tesseract-ocr/langdata_lstm/issues/66) |
| 5 | tesseract-ocr/tesseract | Issue + PR: SIGFPE in `Textord::CleanupSingleRowResult` (float overflow) | [tesseract#4627](https://github.com/tesseract-ocr/tesseract/issues/4627), [tesseract#4628](https://github.com/tesseract-ocr/tesseract/pull/4628), patch `tesseract-cleanup-single-row-overflow.patch` |
| 6 | tesseract-ocr/tesseract | PR: `tessedit_right_to_left` parameter (models with Latin letters are laid out LTR) | [tesseract#4629](https://github.com/tesseract-ocr/tesseract/pull/4629), patch `tesseract-right-to-left-param.patch` |
| 7 | tesseract-ocr/tessdata_contrib | PR: `fas_print`, the fine-tuned Persian model | [tessdata_contrib#19](https://github.com/tesseract-ocr/tessdata_contrib/pull/19); package in `dist/tessdata_contrib/fas_print/` |

---

## 1. tesstrain PR — keep ZWNJ/ZWJ and do not mirror in WordStr labels

**Title:** generate_wordstr_box.py: keep ZWNJ/ZWJ and don't mirror brackets in RTL/Indic labels

`generate_wordstr_box.py` (used for `LANG_TYPE=RTL` and `LANG_TYPE=Indic`)
converts each ground-truth line with `bidi.algorithm.get_display()`. That runs
two Unicode Bidi Algorithm steps that corrupt the training labels:

1. **Rule X9 deletes characters of class BN**, which include U+200C ZWNJ and
   U+200D ZWJ. Persian uses ZWNJ in a large share of words (`می‌خواهم`,
   `کتاب‌ها`), and Indic scripts use ZWJ/ZWNJ to select conjunct/half forms.
   The labels lose them, so a fine-tuned model unlearns them:

   ```
   get_display('شیک‌ترین')  -> 'نیرتکیش'     (ZWNJ gone)
   get_display('क्‍ष')       -> 'क्ष'          (ZWJ gone)
   ```

2. **Rule L4 mirrors paired punctuation** (`()[]{}«»<>`) at RTL levels. But
   Tesseract labels a mirrored glyph with its *logical* code point: with
   `-c hocr_char_boxes=1`, the stock `fas` model reports the bracket at the right
   edge of an RTL `(…)` as U+0028. Mirrored labels teach the model the opposite,
   and after fine-tuning it outputs `)ای‌پی‌ای-۱۱۷(` and `»…«`.

Measured on Persian: fine-tuning `tessdata_best/fas` on the same 20k synthetic
lines for 10k iterations, only the box writer changed. 6,607 of the 19,974
training lines contain ZWNJ; with the current script none of their labels do.
Scored on 998 held-out lines with `--psm 13`:

| labels | CER | WER | ZWNJ emitted (of 407) | `(…)` reversed (of 121 lines) |
|---|---|---|---|---|
| stock `fas`, no fine-tuning | 5.07% | 10.0% | 400 | 0 |
| `get_display()` (current) | 3.67% | 18.0% | 2 | 69 |
| this patch | 2.03% | 3.9% | 402 | 11¹ |

¹ Brackets around Latin words inside RTL text; see below.

The patch replaces `get_display()` with the same UBA steps from python-bidi
minus mirroring, and lets ZWJ/ZWNJ inherit a neighbour's bidi class so they
survive X9 without changing the ordering of anything else. LTR (Indic) lines
come out unchanged except that joiners are kept.

One refinement is not in the patch yet: Tesseract's `ResultIterator` outputs
a word that contains Latin letters and no RTL letters left to right as is, so
brackets inside such a word (`(multiple)` in Persian text) must be labelled by
their drawn (mirrored) shape. Doing that cut reversed brackets around Latin
words from 12/20 to 6/20 lines in our tests; the rest are Latin glued to
Persian without a space, which `ResultIterator` cannot order correctly. Worth
discussing with maintainers whether to include it, since it encodes knowledge
of the recognizer's post-processing in the label writer.

---

## 2. tesstrain PR — create the output dir before writing `all-lstmf`

`make lists` fails on a fresh `DATA_DIR` when the `.lstmf` files already exist:

```
Makefile:243: *** open: .../data/foo/all-lstmf: No such file or directory.  Stop.
```

The `$(ALL_LSTMF)` recipe writes the file with `$(file >$@)`. make expands all
recipe lines before running any of them, so the recipe's own `mkdir -p $(@D)`
runs too late. Fix: add the order-only prerequisite the `$(ALL_GT)` rule
already has:

```make
-$(ALL_LSTMF): $(ALL_FILES:%.gt.txt=%.lstmf)
+$(ALL_LSTMF): $(ALL_FILES:%.gt.txt=%.lstmf) | $(OUTPUT_DIR)
```

---

## 3. Comment on tesseract#4582

> Here is an image-only reproduction, as requested: `issue4582-sahel-200dpi.png`
> (attached; clean 200 dpi render of 14 lines of Persian Wikipedia text in the
> OFL font Sahel). With Tesseract 5.5.3 and `tessdata_best/fas`, the last two
> lines are missing from the output; `-c textord_noise_rejrows=0` restores them.
> `textord_noise_rejwords=0` alone has no effect, so the row filter is the
> culprit.
>
> The impact is larger than a few lines per page. On 998 synthetic single-line
> images (31 fonts, varied size and degradation) with `--psm 7`, **175 lines
> (17.5%) produced empty output**, and dropping them accounts for most of the
> error: CER 20.9% by default vs 5.2% with `textord_noise_rejrows=0` (5.1%
> with `--psm 13`, which skips layout analysis). Larger text is hit more often
> (20% of lines taller than 80 px vs 11% of lines of 30–45 px). On 74 real
> printed Persian pages (PersianML/persian-ocr-benchmark) the flag raises word
> recall from 49.8% to 50.7%.

---

## 4. langdata_lstm issue — problems in `fas` training data

**Title:** fas: Arabic comma forbidden, mixed Arabic/Persian yeh and kaf, bidi controls in training text and wordlist

- `fas/forbidden_characters` contains the range `0x0600-0x0615`, which includes
  U+060C ARABIC COMMA, the standard Persian comma, while `fas/desired_characters`
  lists it. The released `fas` model has no `،` in its unicharset and can never
  output it (cf. tessdata_best#11).
- `fas.training_text`: 249,563 of 788,024 yeh are U+064A ARABIC YEH (Persian
  uses U+06CC), 94,410 of 311,062 kaf are U+0643 (Persian uses U+06A9). The
  model emits the Arabic forms in Persian text.
- `fas.training_text` contains bidi control characters (U+202A–U+202E) and
  `fas.wordlist` has 5,098 entries with them, 27,432 with U+064A and 7,854 with
  U+0643.
- `fas.training_text` has no Latin letters, so the model cannot read the Latin
  words common in Persian documents.
- `okfonts.txt` lists only DejaVu, FreeMono, FreeSerif and Tahoma.

---

## 5. tesseract issue + PR — SIGFPE in `Textord::CleanupSingleRowResult`

**Issue title:** SIGFPE in Textord::CleanupSingleRowResult: float overflow when a row has two unrecognized words (`--psm 7`, fast `fas` model)

**Issue body:**

> **Environment:** Tesseract 5.5.3 (also reproduced with a RelWithDebInfo build
> of the 5.5.3 tag), leptonica 1.87.0, Linux x86-64 with AVX512, `tessdata_fast`
> `fas` (best2int20180322). Deterministic.
>
> **Reproduce** (image attached: a single degraded Persian text line rendered
> from Wikipedia text in an OFL font, no copyrighted content):
>
> ```
> tesseract sigfpe-fas-fast.png - -l fas --psm 7
> Floating point exception (core dumped)      # exit status 136
> ```
>
> `--psm 4`, `6`, `8` and `13` are fine; `-c dotproduct=sse` or `avx` are fine
> too, `auto/generic/native/avx2/fma/avx512` crash. The float `tessdata_best`
> model reads the line without problems.
>
> **Cause:** `Textord::CleanupSingleRowResult` (`src/textord/textord.cpp:275`)
> adds up `word->best_choice->certainty()` for each row in a `float`. A word
> that the recognizer could not classify gets its certainty from
> `WERD_RES::FakeWordFromRatings`, which initializes it to `-FLT_MAX`. On this
> image the row contains two consecutive empty words (`unichar_string_ == ""`,
> certainty `-3.40282347e+38`, seen in gdb), so `-FLT_MAX + -FLT_MAX` overflows
> to `-inf`. `main1()` in `src/tesseract.cpp` enables `FE_OVERFLOW` traps with
> `feenableexcept`, so the overflow raises SIGFPE. The `dotproduct` setting
> only changes which words come out empty, which is why some variants "work".
> Library users (who do not enable the traps) get a silent `-inf` row mean
> instead of a crash.
>
> ```
> #0 tesseract::Textord::CleanupSingleRowResult (...) at src/textord/textord.cpp:275
> #1 tesseract::Tesseract::recog_all_words (...) at src/ccmain/control.cpp:438
> #2 tesseract::TessBaseAPI::Recognize (...) at src/api/baseapi.cpp:837
> #3 tesseract::TessBaseAPI::ProcessPage (...) at src/api/baseapi.cpp:1222
> ```
>
> **Fix:** accumulate the row total in `double` (PR follows). With that change
> the image is recognized normally and 300 synthetic line images that included
> one more crashing line all pass at `--psm 7`. On the first 300 lines of a
> synthetic Persian line set the stock fast model crashes on 1 line in 300, so
> anyone batch-processing line images with `--psm 7` hits this.

**PR title:** Avoid float overflow in Textord::CleanupSingleRowResult

**PR body:** `CleanupSingleRowResult` sums word certainties in `float`. Words
without a classification carry `-FLT_MAX` (set in `WERD_RES::FakeWordFromRatings`),
so a row with two of them overflows to `-inf`, which the CLI turns into SIGFPE
because `main1()` enables `FE_OVERFLOW` traps. Accumulate in `double` instead;
the comparison with `best_conf` is unchanged. Fixes #<issue above>.
Patch: `tesseract-cleanup-single-row-overflow.patch` (tested against the 5.5.3
tag; the image in `repro/sigfpe-fas-fast.png` reads fine afterwards).

---

## 6. tesseract PR — `tessedit_right_to_left`: let a model state its reading direction

**Title:** Add tessedit_right_to_left to override the unicharset majority-direction heuristic

**Body:**

> `Tesseract::init_tesseract_lang_data` sets `right_to_left_` from
> `UNICHARSET::major_right_to_left()`, which counts unichars with bidi class L
> against R/AL/AN. An RTL model that also knows the Latin alphabet, which
> Persian, Arabic, Urdu or Hebrew documents need for the English words and
> product names in them, comes out *left-to-right*: 52 Latin letters outnumber
> the 30–45 letters of the RTL script. Our fine-tuned Persian model
> (`fas_print`, https://github.com/givia/tesseract-fas) has 53 L vs 46 R/AL/AN unichars and is
> therefore laid out LTR by `ColumnFinder` (no `ReflectForRtl`), and
> `PageIterator::Orientation` reports `WRITING_DIRECTION_LEFT_TO_RIGHT` for
> its blocks. Stock `fas` (95 unichars, 1 L) is RTL, so the user-visible
> effect is that the fine-tuned model reads multi-column pages in the wrong
> column order while single-column pages are unaffected.
>
> Measured on three multi-column pages of PersianML/persian-ocr-benchmark
> (relaxed character edit distance to the ground truth, lower is better):
>
> | page | stock `fas` | `fas_print` | `fas_print` + `tessedit_right_to_left=1` |
> |---|---|---|---|
> | image_026, two-column article | 259 | 1,233 | 189 |
> | image_008, exam sheet | 926 | 1,356 | 925 |
> | image_062, phone manual page | 282 | 680 | 278 |
>
> The only workaround with 5.5.x is to list an RTL-majority model first
> (`-l fas+fas_print`), which doubles recognition time and mixes two models'
> output.
>
> This PR adds an integer parameter `tessedit_right_to_left` (default `-1`:
> keep the current heuristic; `0`/`1`: force the direction). A model can set
> it in the config stored in its traineddata, so users need not know about it.
> No behaviour changes unless the parameter is set. Patch:
> `tesseract-right-to-left-param.patch` (three files, 13 lines).
>
> A more general fix would base the decision on the script of the model's
> letters rather than a raw unichar count, but that needs agreement on what
> "primary script" means for mixed models; the parameter is the minimal change
> that unblocks such models today.

---

## 7. tessdata_contrib PR — `fas_print`

**Title:** Add fas_print: Persian (fas) fine-tuned for modern Iranian print

**Body:**

> `fas_print` is `tessdata_best/fas` fine-tuned with Tesseract 5.5.3 on
> 325k synthetic Persian lines (Wikipedia text, 43 fonts including the classic
> Iranian book faces, scan-like degradation) and 1,282 real scanned book lines
> with corrected transcriptions. It fixes the systematic problems of the stock
> model (cannot output the Persian comma, no Latin letters, mixes Arabic and
> Persian yeh/kaf, drops 1 in 6 text lines through the noise filter) and
> reduces the character error rate on real scans by 3–8×:
>
> | test set | stock `fas` | `fas_print` |
> |---|---|---|
> | 190 lines of a 1980s phototypeset book, CER | 5.38% | 0.63% |
> | 321 lines of a 2010s book scan (200 dpi), CER | 4.66% | 1.25% |
> | 4 pages of the Iranian constitution (`--psm 6`), CER | 1.30% | 0.25% |
> | 2,000 synthetic lines in unseen fonts, CER | 5.11% | 1.79% |
> | 74 printed pages of PersianML/persian-ocr-benchmark, word recall | 50.7% | 53.0% |
>
> `best/` is the float model, `fast/` the integer conversion. The README is
> the model card: training data and their licences, evaluation, known
> limitations (multi-column pages are laid out left-to-right until
> tesseract-ocr/tesseract#4629 is merged, workaround given) and reproduction
> (pipeline at https://github.com/givia/tesseract-fas).

Files: `dist/tessdata_contrib/fas_print/{README.md,best/fas_print.traineddata,fast/fas_print.traineddata}`.
