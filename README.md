# ML_treino_eSports

SVC + GridSearchCV que prevê o **vencedor de partidas profissionais de League of Legends** (campeonatos de 2022: LCK, LEC, LCS, CBLOL, Worlds, MSI e outros) usando o estado do jogo aos **15 minutos**.

**Dataset:** Oracle's Elixir 2022 ([Kaggle: LoL E-sports 2022](https://www.kaggle.com/datasets/arthur1511/lol-esports-2022)), 150.588 linhas (12 por partida: 10 jogadores + 2 times).

## Como rodar
```bash
pip install -r requirements.txt
./download_data.sh
python train_svc.py
```

## Decisões de dados
| Ponto | Decisão |
|---|---|
| Granularidade | 1 linha por partida (linha de time, lado azul). Usar as duas linhas de time espelharia a mesma partida em treino e teste. |
| Categóricas | Não entram: só features numéricas aos 10/15 min, então nenhum encode é necessário |
| N/A | Estrutural: jogos `partial` (ex.: LPL) têm 100% de N/A nas stats de timeline, então são descartados. Nos `complete`, 0 N/A. `SimpleImputer` fica no pipeline por robustez. |
| Vazamento | Nenhuma stat de fim de jogo (torres, barões, ouro total). Só snapshots @10/@15 + first blood/dragão/arauto, que nascem aos 5:00 e 8:00. |
| Split | **Temporal, cortado na virada do dia**: treino < 2022-08-10, teste >= 2022-08-10. Séries MD3/MD5 nunca ficam divididas. Asserts no código garantem: partida única, zero partida em treino e teste, zero série dividida. |
| Escala | StandardScaler (SVC é sensível a escala) |

## Resultado
- 10.656 partidas, 33 features numéricas
- Melhor: `kernel=linear, C=0.01` (no interior do grid)
- Acurácia CV 5-fold (treino): **74,6%**
- Acurácia teste (jogos futuros, 2.150): **77,0%** (F1 Red 0,754 / Blue 0,785)
