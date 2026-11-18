"""Compare regularization settings using only the outer training split.

Each validation fold learns its own vocabulary and IDF values. This avoids
leaking validation-document frequencies through a pre-fitted TF-IDF matrix.
Run from the project folder: python src/tune.py
"""

import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from pipeline import load, make_vectorizers, split_data, RES, SEED


CANDIDATES = {
    "Linear SVM": [0.1, 0.25, 0.5, 1],
    "Logistic Regression": [5, 20, 50],
}


def make_estimator(model_name, C):
    _, vectorizer = make_vectorizers()
    if model_name == "Linear SVM":
        model = LinearSVC(C=C, class_weight="balanced", random_state=SEED)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced", random_state=SEED
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("model", model)])


def main():
    df, _ = load()
    train, _ = split_data(df)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for name, candidates in CANDIDATES.items():
        for C in candidates:
            scores = cross_validate(
                make_estimator(name, C), train["doc"], train["label"],
                cv=cv, scoring="f1_macro", n_jobs=1, error_score="raise",
            )
            row = {
                "model": name, "C": C, "n_training_reviews": len(train),
                "mean_macro_f1": float(scores["test_score"].mean()),
                "std_macro_f1": float(scores["test_score"].std()),
                "mean_fit_seconds": float(scores["fit_time"].mean()),
            }
            for fold, score in enumerate(scores["test_score"], start=1):
                row[f"fold_{fold}_macro_f1"] = float(score)
            rows.append(row)
            print(f"{name:22s} C={C:<5g} macroF1={row['mean_macro_f1']:.4f}", flush=True)

    results = pd.DataFrame(rows)
    results["best_for_model"] = False
    best = results.groupby("model")["mean_macro_f1"].idxmax()
    results.loc[best, "best_for_model"] = True
    RES.mkdir(parents=True, exist_ok=True)
    results.to_csv(RES / "tuning.csv", index=False)
    print("Saved training-only cross-validation scores to results/tuning.csv")


if __name__ == "__main__":
    main()
