# Previsão do vencedor em partidas profissionais de LoL (2022) com SVC + GridSearchCV
# Ideia: usar as estatísticas do jogo aos 15 minutos para prever se o time azul ganha
# Dataset: Oracle's Elixir 2022 (Kaggle: arthur1511/lol-esports-2022)

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "2022_LoL_esports_match_data_from_OraclesElixir.csv"
FIG = ROOT / "figures"
TARGET = "result"  # 1 = vitória, 0 = derrota

# objetivos que acontecem antes dos 15 min (dragão nasce aos 5, arauto aos 8)
EARLY_OBJECTIVES = ["firstblood", "firstdragon", "firstherald"]



# ---------------- 1. Carregando os dados ----------------

df_raw = pd.read_csv(DATA, low_memory=False)
print("Dataset original:", df_raw.shape)

# cada partida tem 12 linhas (10 jogadores + 2 times), pego só as linhas de time
times = df_raw[df_raw["position"] == "team"]

# pego só o lado azul para ficar 1 linha por partida
# (se pegar os dois lados a mesma partida aparece duas vezes)
df = times[times["side"] == "Blue"]


# ---------------- 2. Valores nulos ----------------

stats_15 = [c for c in df.columns if c.endswith(("at10", "at15"))]
print("\n% de nulos nas colunas de 10/15 min por tipo de jogo:")
print(df[stats_15].isna().mean(axis=1).groupby(df["datacompleteness"]).mean())

# os jogos "partial" não têm nenhum dado de 10/15 min (100% nulo),
# então não faz sentido preencher, removo esses jogos
df = df[df["datacompleteness"] == "complete"]
df = df.drop_duplicates(subset="gameid")

# ordeno por data para separar treino e teste depois
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print("\nPartidas depois da limpeza:", len(df))
print("Taxa de vitória do azul:", round(df[TARGET].mean(), 3))


# ---------------- 3. Features e target ----------------

# só colunas numéricas, então não precisa de encode
# não uso dados de fim de jogo (torres, barão, ouro total) porque já mostram quem ganhou
features = stats_15 + EARLY_OBJECTIVES
X = df[features].astype(float)
y = df[TARGET].astype(int)

assert TARGET not in X.columns  # garantir que o target não está nas features
print("Features:", X.shape[1], "| nulos restantes:", X.isna().sum().sum())


# ---------------- 4. Treino e teste ----------------

# separo por data: treino com jogos antigos e testo com jogos mais novos
# corto na virada do dia para não separar jogos da mesma série (MD3/MD5)
dia = df["date"].dt.normalize()
dia_corte = dia.iloc[int(len(df) * 0.8)]
treino = (dia < dia_corte).to_numpy()

X_train, X_test = X[treino], X[~treino]
y_train, y_test = y[treino], y[~treino]

# nenhuma partida pode estar no treino e no teste
assert not set(df.loc[treino, "gameid"]) & set(df.loc[~treino, "gameid"])
print(f"\nTreino: {len(X_train)} partidas | Teste: {len(X_test)} partidas (a partir de {dia_corte:%d/%m/%Y})")


# ---------------- 5. Pipeline e GridSearchCV ----------------

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # preenche nulo com a mediana, se tiver
    ("scaler", StandardScaler()),                   # normaliza, o SVC precisa disso
    ("svc", SVC(random_state=SEED)),
])

# C pequeno = modelo mais simples, C grande = pode dar overfitting
param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.001, 0.01, 0.1, 1]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": [0.001, 0.005, 0.01, "scale"]},
]

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print("\nMelhores parâmetros:", grid.best_params_)
print("Acurácia na validação cruzada:", round(grid.best_score_, 4))


# ---------------- 6. Avaliação ----------------

modelo = grid.best_estimator_
acc_treino = accuracy_score(y_train, modelo.predict(X_train))
y_pred = modelo.predict(X_test)
acc_teste = accuracy_score(y_test, y_pred)

# se o treino for bem maior que o teste é sinal de overfitting
print("\nAcurácia treino:", round(acc_treino, 4))
print("Acurácia teste:", round(acc_teste, 4))
print(classification_report(y_test, y_pred, target_names=["Vermelho venceu", "Azul venceu"], digits=4))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

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

# gráficos 1 a 3 vêm dos dados (partidas reais), o 4 vem do modelo

# 1 - quantas vitórias de cada lado
vitorias = y.value_counts().sort_index()
plt.bar(["Vermelho", "Azul"], vitorias.values, color=["red", "blue"])
plt.title("Vitórias por lado")
plt.ylabel("Partidas")
plt.savefig(FIG / "01_vitorias_por_lado.png")
plt.close()

# 2 - taxa de vitória do azul de acordo com a diferença de ouro aos 15 min
faixas = pd.cut(df["golddiffat15"], bins=[-20000, -4000, -2000, -1000, 0, 1000, 2000, 4000, 20000],
                labels=["< -4k", "-4k a -2k", "-2k a -1k", "-1k a 0", "0 a 1k", "1k a 2k", "2k a 4k", "> 4k"])
taxa = df.groupby(faixas, observed=True)[TARGET].mean() * 100
plt.figure(figsize=(8, 4))
plt.bar(taxa.index.astype(str), taxa.values)
plt.axhline(50, color="gray", linestyle="--")
plt.title("Vitória do azul x diferença de ouro aos 15 min")
plt.xlabel("Diferença de ouro (azul - vermelho)")
plt.ylabel("Vitória do azul (%)")
plt.tight_layout()
plt.savefig(FIG / "02_vitoria_por_ouro.png")
plt.close()

# 3 - taxa de vitória do azul quando ele pega o objetivo primeiro
taxa_obj = [df.loc[df[c] == 1, TARGET].mean() * 100 for c in EARLY_OBJECTIVES]
plt.bar(["First blood", "Primeiro dragão", "Primeiro arauto"], taxa_obj)
plt.axhline(50, color="gray", linestyle="--")
plt.title("Vitória do azul quando pega o objetivo primeiro")
plt.ylabel("Vitória (%)")
plt.ylim(0, 100)
plt.savefig(FIG / "03_vitoria_por_objetivo.png")
plt.close()

# 4 - matriz de confusão do SVM no teste
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Vermelho", "Azul"], cmap="Blues")
plt.title("Matriz de confusão - SVM no teste")
plt.savefig(FIG / "04_matriz_confusao.png")
plt.close()

print("\nGráficos salvos em", FIG)
