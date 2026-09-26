# Reproductions for upstream Tesseract issues

All images here are rendered by this project from Persian Wikipedia text
(CC BY-SA) in OFL fonts, so they can be attached to public issues.

## issue4582-sahel-200dpi.png — lines silently dropped (tesseract#4582)

Clean 200 dpi render (Sahel, 11 pt), 14 text lines; ground truth in
`issue4582-sahel-200dpi.gt.txt`.

```sh
tesseract issue4582-sahel-200dpi.png - -l fas                      # last 2 lines missing
tesseract issue4582-sahel-200dpi.png - -l fas \
    -c textord_noise_rejrows=0 -c textord_noise_rejwords=0         # all 14 lines present
```

Observed with Tesseract 5.5.3, leptonica 1.87.0, tessdata_best `fas`
(4.00.00alpha synth20170629). Also reproduced on a Vazirmatn page at 300 dpi
(line 14) and 150 dpi (line 5); not reproduced with Noto Naskh Arabic, Nazli
or Samim at the same settings.

## sigfpe-fas-fast.png — SIGFPE with the fast fas model

A single text line (Parastoo, degraded render). Crashes with the integer
(`tessdata_fast`, `best2int20180322`) model; the float model reads it fine.

```sh
tesseract sigfpe-fas-fast.png - -l fas --psm 7    # Floating point exception, exit 136
```

Backtrace (distribution build, no debug symbols):

```
Thread 1 "tesseract" received signal SIGFPE, Arithmetic exception.
#0  ?? () from /usr/lib/libtesseract.so.5
#1  tesseract::Tesseract::recog_all_words(...) () from /usr/lib/libtesseract.so.5
#2  tesseract::TessBaseAPI::Recognize(tesseract::ETEXT_DESC*) ()
#3  tesseract::TessBaseAPI::ProcessPage(...) ()
```

Tesseract 5.5.3 (Arch `tesseract` package), AVX512 detected. Deterministic.
Depends on the `dotproduct` parameter and page segmentation mode:

| setting | result |
|---|---|
| `-c dotproduct=` auto, generic, native, avx2, fma, avx512 | SIGFPE |
| `-c dotproduct=` sse, avx | OK |
| `--psm 4`, `6`, `8`, `13` | OK |
| `-c load_system_dawg=0 -c load_freq_dawg=0` | SIGFPE (dictionary not involved) |

**Root cause** (RelWithDebInfo build of 5.5.3 under gdb):
`Textord::CleanupSingleRowResult` (`src/textord/textord.cpp:275`) computes the
mean word certainty of each row in `float`. Words that the recognizer could
not classify get their certainty from `WERD_RES::FakeWordFromRatings`, which
initializes it to `-FLT_MAX`. On this image two consecutive empty words have
that certainty; `-FLT_MAX + -FLT_MAX` overflows to `-inf`, and the `tesseract`
CLI enables `FE_OVERFLOW` traps in `main1()` (`src/tesseract.cpp`), so the
overflow becomes SIGFPE. The `dotproduct` variants only change which words
end up empty, which is why some of them "work". Any caller that does not
enable the traps (the C++ API, pytesseract via the library) gets a silent
`-inf` mean instead of a crash.

Fix: accumulate in `double` (`docs/upstream/tesseract-cleanup-single-row-overflow.patch`).
With it, the image is recognized normally and 300 synthetic line images that
included one more crashing line all pass at `--psm 7`. In the same 300 lines
the stock fast model crashed on 1 of 300; on the earlier 998-line synthetic
set the crash rate was similar (a few per thousand lines).
