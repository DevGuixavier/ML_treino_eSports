# ML_treino_eSports

SVC + GridSearchCV para prever o vencedor de um round de **CS:GO** (CT vs T).

**Dataset:** [CS:GO Round Winner Classification (Kaggle)](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification): 122.410 snapshots de partidas profissionais, 96 features (94 numéricas).

## Como rodar
```bash
pip install -r requirements.txt
./download_data.sh
python train_svc.py
```

## Pré-processamento
| Problema | Tratamento |
|---|---|
| Categóricas: `map` (8 valores) | OneHotEncoder (`handle_unknown="ignore"`) |
| `bomb_planted` (bool) | cast para 0/1 |
| Alvo `round_winner` (CT/T) | encode binário (T=1) |
| N/A (0 no dataset) | SimpleImputer no pipeline (mediana / moda), para robustez em produção |
| 4.962 linhas duplicadas | removidas |
| 6 colunas constantes | removidas |
| Escala (SVC é sensível) | StandardScaler |
| Feature engineering | diferenças CT−T (vivos, vida, armadura, dinheiro, capacetes, placar, armas) |

Split estratificado **antes** de qualquer fit (sem vazamento). O GridSearch roda numa amostra de 12k linhas (o SVC escala de forma O(n²)) e a avaliação é num holdout de 20k linhas.

## Resultado
- Melhor: `kernel=rbf, C=10, gamma=0.05`
- Acurácia CV 5-fold: **76,6%**
- Acurácia holdout (20k): **77,0%** (F1 CT 0,761 / T 0,779)
