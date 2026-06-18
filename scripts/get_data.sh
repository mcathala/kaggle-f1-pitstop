#!/usr/bin/env bash
# Refetch the COMPETITION data into data/ — OPTIONAL: it's already committed in this repo.
#
# The external augmentation file (data/f1_strategy_dataset.csv) is committed too and is the
# EXACT revision the model trained on, so it is intentionally NOT downloaded here (the public
# source has since moved to a newer revision). Use this script only to restore the competition
# CSVs, e.g. on a fresh checkout where you chose not to keep them.
#
# Needs Kaggle API creds (see ../.env.example) + a one-time "I Understand and Accept" on the
# competition rules page.
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE="${KAGGLE:-kaggle}"   # set KAGGLE=.venv/bin/kaggle to use the project venv
mkdir -p data

echo "==> Competition data (playground-series-s6e5)"
$KAGGLE competitions download -c playground-series-s6e5 -p data/
unzip -o data/playground-series-s6e5.zip -d data/
rm -f data/playground-series-s6e5.zip

echo
echo "Done -> data/{train,test,sample_submission}.csv"
