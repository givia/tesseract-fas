#!/usr/bin/env bash
# Fine-tune tessdata_best/fas with tesstrain.
#   scripts/train.sh MODEL_NAME GROUND_TRUTH_DIR [extra make variables, e.g. MAX_ITERATIONS=20000]
# Outputs land in work/data/MODEL_NAME/ (checkpoints, training.log) and
# work/data/MODEL_NAME.traineddata.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
model=$1 gt=$(realpath "$2"); shift 2

export PATH="$ROOT/.venv/bin:$PATH"
python3 "$ROOT/scripts/prep_lstmf.py" "$gt"  # .box/.lstmf with correct RTL labels, in parallel
# tesstrain's $(ALL_LSTMF) recipe writes the file with $(file >...), which make
# expands before the recipe's own `mkdir -p`, so the output dir must exist.
mkdir -p "$ROOT/work/data/$model"
# combine_lang_model embeds LANGDATA_DIR/MODEL/MODEL.config into the traineddata.
mkdir -p "$ROOT/work/data/langdata/$model"
cp "$ROOT/data/langdata/model.config" "$ROOT/work/data/langdata/$model/$model.config"

vars=(
  MODEL_NAME="$model" START_MODEL=fas LANG_TYPE=RTL
  TESSDATA="$ROOT/external/tessdata_best"
  DATA_DIR="$ROOT/work/data"
  GROUND_TRUTH_DIR="$gt"
  WORDLIST_FILE="$ROOT/work/langdata/fas.wordlist"
  NUMBERS_FILE="$ROOT/work/langdata/fas.numbers"
  PUNC_FILE="$ROOT/work/langdata/fas.punc"
  "$@"
)
cd "$ROOT/external/tesstrain"
make lists "${vars[@]}"
make training "${vars[@]}"
