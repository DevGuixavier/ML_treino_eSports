#!/usr/bin/env bash
# Fonte: Oracle's Elixir (dados oficiais de partidas profissionais de LoL, 2022)
# Opção 1 (Kaggle): kaggle datasets download -d arthur1511/lol-esports-2022 -p data --unzip
# Opção 2 (mirror GitHub LFS do CSV original):
set -euo pipefail
mkdir -p data
F=data/2022_LoL_esports_match_data_from_OraclesElixir.csv
curl -sSL -o "$F" \
  "https://media.githubusercontent.com/media/Samuel-Kelly-hub/Esports-Results-Project/HEAD/data/2022_LoL_esports_match_data_from_OraclesElixir.csv"
echo "13b811948dfc2fdaa52d19e3fd085a48088e2d4f68af53792ac684e49e063507  $F" | sha256sum -c -
