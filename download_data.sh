#!/usr/bin/env bash
# Opção 1 (oficial): kaggle datasets download -d christianlillelund/csgo-round-winner-classification -p data --unzip
# Opção 2 (mirror GitHub LFS do mesmo arquivo, sha256 e9851010...3827):
set -euo pipefail
mkdir -p data
curl -sSL -o data/csgo_round_snapshots.csv \
  "https://media.githubusercontent.com/media/grv08singh/01_masterRepo/main/02_EPGC_Intellipaat/01%20EPGC%20-%20Live%20Classes/2025.08.24%20-%20EPGC%20ML%20-%20ML%20using%20PyCaret%20HandsOn/csgo_round_snapshots.csv"
echo "e985101012756f365b4003ebbbb16012f5f4c513b83e1999545f618183113827  data/csgo_round_snapshots.csv" | sha256sum -c -
