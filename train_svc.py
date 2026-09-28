"""
CS:GO Round Winner - SVC + GridSearchCV
Dataset: Kaggle "CS:GO Round Winner Classification" (christianlillelund)
          arquivo csgo_round_snapshots.csv (122.410 snapshots, 97 colunas)
Alvo: round_winner (CT=0 / T=1)
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 42
DATA = Path(__file__).parent / "data" / "csgo_round_snapshots.csv"
TARGET = "round_winner"
# SVC é O(n²)~O(n³) em memória/tempo: busca em amostra, avaliação em holdout grande
N_GRID = 12_000
N_TEST = 20_000


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Diferenças CT - T: o que decide o round é a vantagem relativa, não o absoluto."""
    X = X.copy()
    for a in ["players_alive", "health", "armor", "money", "helmets", "score"]:
        X[f"diff_{a}"] = X[f"ct_{a}"] - X[f"t_{a}"]
    ct_w = [c for c in X.columns if c.startswith("ct_weapon_")]
    t_w = [c for c in X.columns if c.startswith("t_weapon_")]
    X["diff_weapons"] = X[ct_w].sum(axis=1) - X[t_w].sum(axis=1)
    return X


def load() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA)
    print(f"bruto: {df.shape} | N/A: {df.isna().sum().sum()} | duplicadas: {df.duplicated().sum()}")

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])  # alvo nulo não pode ser imputado

    # colunas constantes não carregam informação (ex.: armas nunca usadas)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    print(f"limpo: {df.shape} | constantes removidas: {const}")

    y = (df.pop(TARGET) == "T").astype(int)          # encode binário do alvo
    df["bomb_planted"] = df["bomb_planted"].astype(int)  # bool -> 0/1
    return add_features(df), y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = ["map"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),  # robustez a N/A em produção
            ("sc", StandardScaler()),                   # SVC é sensível a escala
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("svc", SVC(random_state=SEED))])


def main():
    X, y = load()

    # holdout estratificado ANTES de qualquer fit -> sem vazamento de scaler/encoder
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=N_TEST, stratify=y, random_state=SEED)
    X_grid, _, y_grid, _ = train_test_split(
        X_rest, y_rest, train_size=N_GRID, stratify=y_rest, random_state=SEED)

    param_grid = [
        {"svc__kernel": ["rbf"], "svc__C": [1, 10, 50], "svc__gamma": ["scale", 0.01, 0.05]},
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1]},
    ]
    grid = GridSearchCV(
        build_pipeline(X), param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1, verbose=1,
    )
    grid.fit(X_grid, y_grid)

    print(f"\nmelhores params: {grid.best_params_}")
    print(f"acurácia CV (5-fold): {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print(f"acurácia holdout ({len(y_test)} amostras nunca vistas): {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["CT", "T"], digits=4))
    print("matriz de confusão:\n", confusion_matrix(y_test, y_pred))

    out = Path(__file__).parent / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(grid.best_estimator_, out / "svc_csgo.joblib")
    pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[
        ["params", "mean_test_score", "std_test_score", "mean_fit_time"]
    ].to_csv(out / "grid_results.csv", index=False)


if __name__ == "__main__":
    main()
