"""
LoL Esports 2022 (campeonatos profissionais) - SVC + GridSearchCV
Dataset: Oracle's Elixir "2022_LoL_esports_match_data_from_OraclesElixir.csv"
         (espelhado no Kaggle: arthur1511/lol-esports-2022)
Pergunta: com o estado do jogo aos 15 minutos, o time azul vence a partida?
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "2022_LoL_esports_match_data_from_OraclesElixir.csv"
TARGET = "result"
# eventos que, pelas regras de 2022, ocorrem antes dos 15 min
# (1º dragão nasce aos 5:00, arauto aos 8:00). Nada de stats de fim de jogo.
EARLY_OBJECTIVES = ["firstblood", "firstdragon", "firstherald"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    print(f"bruto: {df.shape} (10 linhas de jogador + 2 de time por partida)")

    # 1 linha por partida: linha de TIME do lado azul. Se o lado vermelho também
    # entrasse, a mesma partida apareceria espelhada em treino e teste (vazamento).
    df = df[(df["position"] == "team") & (df["side"] == "Blue")]

    # N/A estrutural: jogos 'partial' (ex.: LPL) não têm NENHUMA stat de timeline.
    # Imputar 100% das features de uma linha é inventar dado, então descartamos.
    na_by_completeness = df.filter(like="at15").isna().mean(axis=1).groupby(df["datacompleteness"]).mean()
    print(f"fração de N/A nas stats @15 por completude:\n{na_by_completeness.to_string()}")
    df = df[df["datacompleteness"] == "complete"]

    df = df.drop_duplicates(subset="gameid")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"partidas usadas: {len(df)} | vitória azul: {df[TARGET].mean():.3f}")
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    # só colunas numéricas medidas aos 10/15 min -> nenhuma categórica, nenhum encode necessário
    snap = [c for c in df.columns if c.endswith(("at10", "at15"))]
    return df[snap + EARLY_OBJECTIVES].astype(float)


def main():
    df = load()
    X, y = features(df), df[TARGET].astype(int)
    print(f"features: {X.shape[1]} numéricas | N/A restantes: {int(X.isna().sum().sum())}")

    # split TEMPORAL cortando na virada do DIA: jogos de uma mesma série (MD3/MD5)
    # acontecem no mesmo dia e nunca ficam divididos entre treino e teste
    day = df["date"].dt.normalize()
    cut_day = day.iloc[int(len(df) * 0.8)]
    train = (day < cut_day).to_numpy()
    X_train, X_test, y_train, y_test = X[train], X[~train], y[train], y[~train]

    assert df["gameid"].is_unique, "partida duplicada"
    assert not set(df.loc[train, "gameid"]) & set(df.loc[~train, "gameid"]), "partida em treino e teste"
    assert day[train].max() < day[~train].min(), "série dividida entre treino e teste"
    print(f"treino < {cut_day:%Y-%m-%d} ({train.sum()}) | teste >= {cut_day:%Y-%m-%d} ({(~train).sum()})")

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),  # robustez a N/A pontual em produção
        ("sc", StandardScaler()),                   # SVC é sensível a escala
        ("svc", SVC(random_state=SEED)),
    ])
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.001, 0.01, 0.1, 1]},
        {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": [0.001, 0.005, 0.01, "scale"]},
    ]
    grid = GridSearchCV(
        pipe, param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\nmelhores params: {grid.best_params_}")
    print(f"acurácia CV (5-fold): {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print(f"acurácia teste (jogos futuros): {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Red vence", "Blue vence"], digits=4))
    print("matriz de confusão:\n", confusion_matrix(y_test, y_pred))

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_, out / "svc_lol_esports.joblib")
    pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[
        ["params", "mean_test_score", "std_test_score", "mean_fit_time"]
    ].to_csv(out / "grid_results.csv", index=False)


if __name__ == "__main__":
    main()
