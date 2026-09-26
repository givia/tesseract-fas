#!/usr/bin/env bash
# Download everything the pipeline needs into external/ (gitignored).
# Safe to re-run: existing files are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
EXT=external
mkdir -p "$EXT"/{tessdata_best,langdata/fas,fonts,benchmark/images}

get() { # url dest
  [[ -s "$2" ]] && return 0
  echo "fetch $2"
  curl -gfsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

# --- Tesstrain -----------------------------------------------------------
[[ -d $EXT/tesstrain ]] || git clone -q --depth 1 https://github.com/tesseract-ocr/tesstrain "$EXT/tesstrain"

# --- Start models (float, required for fine-tuning) ----------------------
for m in fas eng; do
  get "https://github.com/tesseract-ocr/tessdata_best/raw/main/$m.traineddata" "$EXT/tessdata_best/$m.traineddata"
done

# --- Google's original Persian langdata ----------------------------------
for f in fas.training_text fas.wordlist fas.punc fas.numbers fas.unicharset \
         desired_characters forbidden_characters okfonts.txt; do
  get "https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/fas/$f" "$EXT/langdata/fas/$f"
done

# --- Fonts (release zips; only the TTFs are kept) ------------------------
fonts=(
  "vazirmatn  https://github.com/rastikerdar/vazirmatn/releases/download/v33.003/vazirmatn-v33.003.zip"
  "sahel      https://github.com/rastikerdar/sahel-font/releases/download/v3.4.0/sahel-font-v3.4.0.zip"
  "samim      https://github.com/rastikerdar/samim-font/releases/download/v4.0.5/samim-font-v4.0.5.zip"
  "shabnam    https://github.com/rastikerdar/shabnam-font/releases/download/v5.0.1/shabnam-font-v5.0.1.zip"
  "tanha      https://github.com/rastikerdar/tanha-font/releases/download/v0.10/tanha-font-v0.10.zip"
  "parastoo   https://github.com/rastikerdar/parastoo-font/releases/download/v2.0.1/parastoo-font-v2.0.1.zip"
  "gandom     https://github.com/rastikerdar/gandom-font/releases/download/v0.8/gandom-font-v0.8.zip"
  "nahid      https://github.com/rastikerdar/nahid-font/releases/download/v0.3.0/nahid-font-v0.3.0.zip"
  "estedad    https://github.com/aminabedi68/Estedad/releases/download/8.5/Estedad-v8.5.zip"
)
for entry in "${fonts[@]}"; do
  read -r name url <<<"$entry"
  dir="$EXT/fonts/$name"
  [[ -d $dir ]] && continue
  tmp=$(mktemp -d)
  curl -fsSL --retry 3 -o "$tmp/f.zip" "$url"
  mkdir -p "$dir"
  unzip -q -j -o "$tmp/f.zip" '*.ttf' -d "$dir" 2>/dev/null || true
  rm -rf "$tmp"
  echo "font $name: $(ls "$dir" | wc -l) ttf"
done

# Classic Persian print faces (Zar, Nazli, Roya, Titr, ...), GPL-2, via Debian.
if [[ ! -d $EXT/fonts/farsiweb ]]; then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/f.deb" https://deb.debian.org/debian/pool/main/f/fonts-farsiweb/fonts-farsiweb_0.4.dfsg-13_all.deb
  (cd "$tmp" && bsdtar -xf f.deb && bsdtar -xf data.tar.*)
  mkdir -p "$EXT/fonts/farsiweb"
  find "$tmp" -name '*.ttf' -exec cp {} "$EXT/fonts/farsiweb/" \;
  rm -rf "$tmp"
  echo "font farsiweb: $(ls "$EXT/fonts/farsiweb" | wc -l) ttf"
fi

# Naskh book faces from Google Fonts (OFL).
GF=https://github.com/google/fonts/raw/main/ofl
for f in amiri/Amiri-Regular amiri/Amiri-Bold scheherazadenew/ScheherazadeNew-Regular \
         scheherazadenew/ScheherazadeNew-Bold mirza/Mirza-Regular mirza/Mirza-Bold; do
  mkdir -p "$EXT/fonts/googlefonts"
  get "$GF/$f.ttf" "$EXT/fonts/googlefonts/$(basename "$f").ttf"
done

# Classic Iranian book/office faces ("B" fonts, Borna Rayaneh freeware; licence
# unclear, used only to render training images locally, never redistributed).
BF=https://github.com/intuxicated/css-persian/raw/master/fonts
mkdir -p "$EXT/fonts/bfonts"
for f in BLotus BMitra BNazanin BZar BBadr BYagut BRoya BCompset BTraffic BYekan BTitrBold \
         BKoodakBold BYas; do
  get "$BF/$f.ttf" "$EXT/fonts/bfonts/$f.ttf"
done

# --- Training text: one shard of Persian Wikipedia (CC BY-SA) -----------
mkdir -p "$EXT/corpus"
get "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.fa/train-00001-of-00004.parquet" \
    "$EXT/corpus/fawiki-1.parquet"

# --- Real-world evaluation set (evaluation only; do not redistribute) ----
HF=https://huggingface.co/datasets/PersianML/persian-ocr-benchmark/resolve/main/bench_data
get "$HF/ocr_eval.jsonl" "$EXT/benchmark/ocr_eval.jsonl"
python3 - "$EXT/benchmark" "$HF" <<'EOF'
import json, sys, subprocess, pathlib
root, hf = pathlib.Path(sys.argv[1]), sys.argv[2]
for line in open(root / "ocr_eval.jsonl", encoding="utf-8-sig"):
    img = json.loads(line)["image"]            # e.g. images/image_001.jpg
    dest = root / img
    if not dest.exists():
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(dest), f"{hf}/{img}"], check=True)
EOF
echo "benchmark: $(ls "$EXT/benchmark/images" | wc -l) images"
