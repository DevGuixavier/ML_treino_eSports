# Previsão do vencedor em partidas profissionais de LoL (2022) com SVC + GridSearchCV
# Ideia: usar as estatísticas do jogo aos 15 minutos para prever se o time azul ganha
# Dataset: Oracle's Elixir 2022 (Kaggle: arthur1511/lol-esports-2022)

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, validation_curve
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

# cores dos gráficos
AZUL, VERMELHO, LARANJA = "#2a78d6", "#e34948", "#eb6834"
TEXTO, TEXTO2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": TEXTO2, "text.color": TEXTO,
    "xtick.color": TEXTO2, "ytick.color": TEXTO2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def salvar(nome):
    plt.tight_layout()
    plt.savefig(FIG / nome, dpi=150)
    plt.close()


# 1 - balanceamento do target
cont = y.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar(["Vermelho venceu", "Azul venceu"], cont.values, color=[VERMELHO, AZUL], width=0.5)
ax.bar_label(barras, labels=[f"{v} ({v / len(y):.1%})" for v in cont.values], padding=4)
ax.set_title("Distribuição do target")
ax.set_ylabel("Partidas")
ax.grid(axis="x", visible=False)
salvar("01_distribuicao_target.png")

# 2 - nulos por tipo de jogo
t_azul = times[times["side"] == "Blue"]
nulos = t_azul[stats_15].isna().mean(axis=1).groupby(t_azul["datacompleteness"]).mean() * 100
qtd = t_azul["datacompleteness"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar(nulos.index, nulos.values, color=AZUL, width=0.5)
ax.bar_label(barras, labels=[f"{v:.0f}% nulo\n({qtd[i]} jogos)" for i, v in nulos.items()], padding=4)
ax.set_title("% de nulos nas stats de 10/15 min")
ax.set_ylabel("% de nulos")
ax.set_ylim(0, 120)
ax.grid(axis="x", visible=False)
salvar("02_nulos_por_completude.png")

# 3 - correlação das features com o target
corr = X.corrwith(y).sort_values(key=abs, ascending=False).head(15).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(corr.index, corr.values, color=[AZUL if v > 0 else VERMELHO for v in corr.values], height=0.6)
ax.axvline(0, color=TEXTO2, linewidth=1)
ax.set_title("Top 15 features por correlação com a vitória")
ax.set_xlabel("Correlação com o target (azul = ajuda a ganhar, vermelho = atrapalha)")
ax.grid(axis="y", visible=False)
salvar("03_correlacao_features.png")

# 4 - taxa de vitória pela diferença de ouro aos 15 min
faixas = pd.cut(df["golddiffat15"], bins=np.arange(-8000, 8001, 1000))
taxa = df.groupby(faixas, observed=True)[TARGET].agg(["mean", "size"])
taxa = taxa[taxa["size"] >= 30]  # ignoro faixas com poucos jogos
centro = [f.mid for f in taxa.index]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(centro, taxa["mean"] * 100, color=AZUL, linewidth=2, marker="o", markersize=6)
ax.axhline(50, color=TEXTO2, linewidth=1, linestyle="--")
ax.axvline(0, color=TEXTO2, linewidth=1, linestyle="--")
ax.set_title("Taxa de vitória do azul x diferença de ouro aos 15 min")
ax.set_xlabel("Diferença de ouro aos 15 min (azul - vermelho)")
ax.set_ylabel("Vitória do azul (%)")
ax.set_ylim(0, 100)
salvar("04_vitoria_por_ouro.png")

# 5 - curva de validação do C (para ver overfitting)
valores_c = [0.0001, 0.001, 0.01, 0.1, 1]
pipe_linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                        ("svc", SVC(kernel="linear", random_state=SEED))])
tr_scores, cv_scores = validation_curve(
    pipe_linear, X_train, y_train, param_name="svc__C", param_range=valores_c,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(valores_c, tr_scores.mean(axis=1) * 100, color=AZUL, linewidth=2, marker="o", label="Treino")
ax.plot(valores_c, cv_scores.mean(axis=1) * 100, color=LARANJA, linewidth=2, marker="o", label="Validação")
ax.set_xscale("log")
ax.set_title("Curva de validação do C (SVC linear)")
ax.set_xlabel("C (escala log)")
ax.set_ylabel("Acurácia (%)")
ax.legend(frameon=False)
salvar("05_curva_validacao_C.png")

# 6 - matriz de confusão
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else TEXTO)
ax.set_xticks([0, 1], ["Vermelho", "Azul"])
ax.set_yticks([0, 1], ["Vermelho", "Azul"])
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusão (acurácia {acc_teste:.1%})")
ax.grid(False)
salvar("06_matriz_confusao.png")

# 7 - treino x validação x teste
fig, ax = plt.subplots(figsize=(6, 4))
nomes = ["Treino", "Validação cruzada", "Teste"]
vals = [acc_treino * 100, grid.best_score_ * 100, acc_teste * 100]
barras = ax.bar(nomes, vals, color=AZUL, width=0.5)
ax.bar_label(barras, labels=[f"{v:.1f}%" for v in vals], padding=4)
ax.set_ylim(0, 100)
ax.set_title("Acurácia do modelo final")
ax.set_ylabel("Acurácia (%)")
ax.grid(axis="x", visible=False)
salvar("07_acuracia_final.png")

print("\nGráficos salvos em", FIG)
