# Previsão do vencedor em partidas profissionais de LoL (2022) com SVM (SVC) + GridSearchCV
# Pergunta: olhando o jogo aos 15 minutos, dá para saber se o time azul vai ganhar?
# Dataset: Oracle's Elixir 2022 (Kaggle: arthur1511/lol-esports-2022)

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "2022_LoL_esports_match_data_from_OraclesElixir.csv"
FIG = ROOT / "figures"
TARGET = "result"  # 1 = azul venceu, 0 = vermelho venceu


# ---------------- 1. Carregando os dados ----------------
# O dataset tem 12 linhas por partida: 10 jogadores + 2 linhas de time (azul e vermelho).
# Uso só a linha do time azul, assim fica 1 linha por partida.
# Se usasse as duas, a mesma partida apareceria duas vezes e poderia cair no treino e no teste.

df_raw = pd.read_csv(DATA, low_memory=False)
print("Dataset original:", df_raw.shape)

df = df_raw[(df_raw["position"] == "team") & (df_raw["side"] == "Blue")]


# ---------------- 2. Valores nulos ----------------
# Os jogos "partial" (principalmente das ligas chinesas LPL e LDL) não têm nenhum dado
# de 10/15 min, é 100% nulo.
# Não dá para preencher uma linha inteira sem inventar o jogo, então removo esses jogos.
# Nos jogos "complete" não sobra nenhum nulo.

stats_15 = [c for c in df.columns if c.endswith(("at10", "at15"))]
print("\n% de nulos nas stats de 10/15 min por tipo de jogo:")
print(df[stats_15].isna().mean(axis=1).groupby(df["datacompleteness"]).mean())

df = df[df["datacompleteness"] == "complete"].drop_duplicates(subset="gameid").copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print("\nPartidas depois da limpeza:", len(df))
print("Vitórias do azul:", f"{df[TARGET].mean():.1%}", "-> classes equilibradas, não precisa balancear")


# ---------------- 3. Features ----------------
# Features = o que o modelo "vê" para decidir.
# Uso só a DIFERENÇA entre os times (azul - vermelho) aos 10 e 15 min:
#   positivo = azul está na frente, negativo = vermelho está na frente.
# Colunas como goldat15 e opp_goldat15 separadas repetem a informação de golddiffat15,
# então uso só as diferenças.
# Só uso colunas medidas exatamente aos 10 e 15 min. Não uso first blood, primeiro dragão
# e primeiro arauto porque o dataset não diz QUANDO aconteceram (podem ter sido depois dos 15 min).
# Também não uso nada de fim de jogo (torres, barão, ouro final) porque isso já mostra quem ganhou.

df["killsdiffat10"] = df["killsat10"] - df["opp_killsat10"]
df["killsdiffat15"] = df["killsat15"] - df["opp_killsat15"]

features = [
    "golddiffat10", "xpdiffat10", "csdiffat10", "killsdiffat10",
    "golddiffat15", "xpdiffat15", "csdiffat15", "killsdiffat15",
]
X = df[features].astype(float)
y = df[TARGET].astype(int)
assert TARGET not in X.columns  # o resultado não pode estar nas features


# ---------------- 4. Treino e teste ----------------
# Separo por data: treino com os jogos mais antigos e teste com os mais novos,
# como se o modelo estivesse prevendo jogos que ainda não aconteceram.
# Corto na virada do dia para não separar jogos da mesma série (MD3/MD5).

dia = df["date"].dt.normalize()
dia_corte = dia.iloc[int(len(df) * 0.8)]
treino = (dia < dia_corte).to_numpy()

X_train, X_test = X[treino], X[~treino]
y_train, y_test = y[treino], y[~treino]
assert not set(df.loc[treino, "gameid"]) & set(df.loc[~treino, "gameid"])
print(f"\nTreino: {len(X_train)} partidas | Teste: {len(X_test)} partidas (a partir de {dia_corte:%d/%m/%Y})")


# ---------------- 5. Modelo: SVM + GridSearchCV ----------------
# O SVM procura a "linha" (hiperplano) que melhor separa vitórias do azul e do vermelho.
# Ele decide pela distância de cada partida até essa linha, por isso precisa normalizar:
# sem o StandardScaler, o ouro (milhares) pesaria muito mais que os abates (unidades).
#
# O GridSearchCV testa combinações de parâmetros e escolhe a de maior acurácia média
# em 5 partes do treino (validação cruzada):
#   kernel linear = fronteira reta | kernel rbf = fronteira curva
#   C pequeno = modelo mais simples | C grande = se ajusta mais ao treino (risco de overfitting)
#   gamma (rbf) = o quanto a curva pode "entortar"

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # não há nulos, mas protege dados novos
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=SEED)),
])

param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.001, 0.01, 0.1, 1]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": [0.001, 0.01, "scale"]},
]

grid = GridSearchCV(
    pipeline, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="accuracy", n_jobs=-1, verbose=1,
)
grid.fit(X_train, y_train)
modelo = grid.best_estimator_

print("\nMelhores parâmetros:", grid.best_params_)
print(f"Acurácia na validação cruzada: {grid.best_score_:.1%}")


# ---------------- 6. Resultado ----------------
# Overfitting: se a acurácia no treino for bem maior que no teste, o modelo decorou o treino.

y_pred = modelo.predict(X_test)
acc_treino = accuracy_score(y_train, modelo.predict(X_train))
acc_teste = accuracy_score(y_test, y_pred)

print(f"\nAcurácia no treino: {acc_treino:.1%}")
print(f"Acurácia no teste:  {acc_teste:.1%}")
print(classification_report(y_test, y_pred, target_names=["Vermelho venceu", "Azul venceu"], digits=3))

# salvando modelo, resultados do grid e dados tratados
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(modelo, ROOT / "models" / "svc_lol_esports.joblib")
pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[
    ["params", "mean_test_score", "std_test_score"]
].to_csv(ROOT / "models" / "grid_results.csv", index=False)
df[["gameid", "date", "league", "teamname"] + features + [TARGET]].to_csv(
    ROOT / "data" / "partidas_tratadas.csv", index=False)


# ---------------- 7. Gráficos ----------------

FIG.mkdir(exist_ok=True)
for antigo in FIG.glob("*.png"):
    antigo.unlink()

# 1 - (dados) distribuição do target
# Interpretação: as duas classes têm quantidades parecidas, então não precisa balancear.
vitorias = y.value_counts().sort_index()
plt.bar(["Vermelho venceu", "Azul venceu"], vitorias.values, color=["red", "blue"])
for i, v in enumerate(vitorias.values):
    plt.text(i, v + 50, f"{v} ({v / len(y):.1%})", ha="center")
plt.title("Distribuição do target")
plt.ylabel("Partidas")
plt.savefig(FIG / "01_distribuicao_target.png")
plt.close()

# 2 - (dados) distribuição da diferença de ouro aos 15 min, separada por quem venceu
# Interpretação: quando o azul vence, a diferença fica mais à direita (positiva);
# quando perde, fica mais à esquerda. A parte onde as cores se misturam são os jogos
# parelhos, que é onde o modelo mais erra.
plt.figure(figsize=(8, 4))
plt.hist(df.loc[y == 0, "golddiffat15"], bins=50, alpha=0.6, color="red", label="Vermelho venceu")
plt.hist(df.loc[y == 1, "golddiffat15"], bins=50, alpha=0.6, color="blue", label="Azul venceu")
plt.axvline(0, color="black", linestyle="--")
plt.title("Distribuição da diferença de ouro aos 15 min")
plt.xlabel("Diferença de ouro (azul - vermelho)")
plt.ylabel("Partidas")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "02_distribuicao_ouro.png")
plt.close()

# 3 - (modelo) matriz de confusão
# Interpretação: diagonal = acertos, fora da diagonal = erros.
# Erros parecidos dos dois lados = o modelo não favorece nenhum time.
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Vermelho", "Azul"], cmap="Blues")
plt.title("Matriz de confusão - SVM no teste")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig(FIG / "03_matriz_confusao.png")
plt.close()

print("\nGráficos salvos em", FIG)
