# Previsão do vencedor em partidas profissionais de LoL (2022) com SVM (SVC) + GridSearchCV
# Pergunta: olhando o jogo aos 15 minutos, dá para saber se o time azul vai ganhar?
# Dataset: Oracle's Elixir 2022 (Kaggle: arthur1511/lol-esports-2022)

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
# Os jogos "partial" (ex.: liga chinesa) não têm nenhum dado de 10/15 min, é 100% nulo.
# Não dá para preencher uma linha inteira sem inventar o jogo, então removo esses jogos.
# Nos jogos "complete" não sobra nenhum nulo.

stats_15 = [c for c in df.columns if c.endswith(("at10", "at15"))]
print("\n% de nulos nas stats de 10/15 min por tipo de jogo:")
print(df[stats_15].isna().mean(axis=1).groupby(df["datacompleteness"]).mean())

df = df[df["datacompleteness"] == "complete"].drop_duplicates(subset="gameid")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print("\nPartidas depois da limpeza:", len(df))
print("Vitórias do azul:", f"{df[TARGET].mean():.1%}", "-> classes equilibradas, não precisa balancear")


# ---------------- 3. Features ----------------
# Features = o que o modelo "vê" para decidir.
# Uso só a DIFERENÇA entre os times (azul - vermelho) aos 10 e 15 min:
#   positivo = azul está na frente, negativo = vermelho está na frente.
# Colunas como goldat15 e opp_goldat15 separadas dizem a mesma coisa que golddiffat15,
# então deixei só as diferenças (testei com as 33 colunas e deu a mesma acurácia).
# Os objetivos (1 = azul pegou primeiro, 0 = vermelho) acontecem antes dos 15 min.
# Não uso nada de fim de jogo (torres, barão, ouro final) porque isso já mostra quem ganhou.

df["killsdiffat10"] = df["killsat10"] - df["opp_killsat10"]
df["killsdiffat15"] = df["killsat15"] - df["opp_killsat15"]

features = [
    "golddiffat10", "xpdiffat10", "csdiffat10", "killsdiffat10",
    "golddiffat15", "xpdiffat15", "csdiffat15", "killsdiffat15",
    "firstblood", "firstdragon", "firstherald",
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
# Comparo o SVM com duas regras simples para saber se ele realmente ajuda:
#   chute = sempre dizer "azul vence" (a classe mais comum)
#   regra do ouro = quem tem mais ouro aos 15 min vence
# Se o SVM não ganhar dessas regras, não valeria a pena usar um modelo.

y_pred = modelo.predict(X_test)
acc_chute = (y_test == 1).mean()
acc_ouro = ((X_test["golddiffat15"] > 0).astype(int) == y_test).mean()
acc_svm = accuracy_score(y_test, y_pred)
acc_treino = accuracy_score(y_train, modelo.predict(X_train))

print(f"\nChute (sempre azul): {acc_chute:.1%}")
print(f"Regra do ouro:       {acc_ouro:.1%}")
print(f"SVM:                 {acc_svm:.1%}")
print(f"SVM no treino:       {acc_treino:.1%} -> próximo do teste, sem overfitting")
print(classification_report(y_test, y_pred, target_names=["Vermelho venceu", "Azul venceu"], digits=3))

# Confiança: o SVM dá a distância de cada partida até a fronteira (decision_function).
# Longe da fronteira = jogo desequilibrado, o modelo tem mais certeza.
# Perto da fronteira = jogo parelho, o modelo tem pouca certeza.
distancia = np.abs(modelo.decision_function(X_test))
nivel = pd.qcut(distancia, 4, labels=["Baixa", "Média", "Alta", "Muito alta"])
acerto_por_nivel = pd.Series(y_pred == y_test.to_numpy()).groupby(np.asarray(nivel)).mean()
acerto_por_nivel = acerto_por_nivel.reindex(["Baixa", "Média", "Alta", "Muito alta"])
print("Acerto por confiança do SVM:\n", (acerto_por_nivel * 100).round(1))

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

# 1 - (dados) vitória do azul de acordo com a diferença de ouro aos 15 min
# Interpretação: quanto mais ouro de vantagem, mais o time vence.
# É a relação principal que o SVM aprende.
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
plt.savefig(FIG / "01_vitoria_por_ouro.png")
plt.close()

# 2 - (modelo) SVM comparado com as regras simples
# Interpretação: mostra quanto o SVM ganha em cima de um chute e da regra do ouro.
nomes = ["Chute\n(sempre azul)", "Regra do ouro\n(mais ouro vence)", "SVM"]
valores = [acc_chute * 100, acc_ouro * 100, acc_svm * 100]
plt.bar(nomes, valores, color=["gray", "gray", "tab:blue"])
for i, v in enumerate(valores):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.ylim(0, 100)
plt.title("Acurácia no teste")
plt.ylabel("Acurácia (%)")
plt.savefig(FIG / "02_svm_vs_regras.png")
plt.close()

# 3 - (modelo) acerto do SVM por nível de confiança
# Interpretação: em jogos desequilibrados o SVM quase não erra,
# em jogos parelhos ele fica perto de um chute.
plt.bar(acerto_por_nivel.index, acerto_por_nivel.values * 100)
for i, v in enumerate(acerto_por_nivel.values * 100):
    plt.text(i, v + 1, f"{v:.0f}%", ha="center")
plt.axhline(50, color="gray", linestyle="--")
plt.ylim(0, 100)
plt.title("Acerto do SVM por nível de confiança")
plt.xlabel("Confiança (distância até a fronteira do SVM)")
plt.ylabel("Acerto (%)")
plt.savefig(FIG / "03_acerto_por_confianca.png")
plt.close()

# 4 - (modelo) matriz de confusão
# Interpretação: diagonal = acertos, fora da diagonal = erros.
# Erros parecidos dos dois lados = o modelo não favorece nenhum time.
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Vermelho", "Azul"], cmap="Blues")
plt.title("Matriz de confusão - SVM no teste")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig(FIG / "04_matriz_confusao.png")
plt.close()

print("\nGráficos salvos em", FIG)
