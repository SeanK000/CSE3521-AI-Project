"""CSE 3521 Milestone 2 - feature pipeline, classical models, and evaluation.

Run from the repo root:
    python src/pipeline.py
Evaluation outputs go to results/; fitted models go to models/.
"""
import html
import json
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

SEED = 47
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "group_sample_seed47.csv"
RES = ROOT / "results"
FIG = ROOT / "figures"
ARTIFACTS = ROOT / "models" / "artifacts.joblib"
LABELS = ["negative", "neutral", "positive"]


def star_to_label(s):
    if s not in (1, 2, 3, 4, 5):
        raise ValueError(f"Expected an integer star rating from 1 to 5, got {s!r}")
    return "negative" if s <= 2 else ("neutral" if s == 3 else "positive")


def clean(text):
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load():
    """Validate the shared sample and remove repeated cleaned review bodies.

    Keep the first occurrence, even when duplicate bodies have conflicting
    labels. Record those conflicts rather than silently assigning a new label.
    """
    df = pd.read_csv(DATA)
    n_raw = len(df)
    required = {"Id", "Score", "Summary", "Text"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

    quality = {
        "missing_text": int(df["Text"].isna().sum()),
        "missing_summary": int(df["Summary"].isna().sum()),
    }
    scores = pd.to_numeric(df["Score"], errors="coerce")
    valid_scores = scores.isin([1, 2, 3, 4, 5])
    quality["invalid_scores_removed"] = int((~valid_scores).sum())
    df = df.loc[valid_scores].copy()
    df["Score"] = scores.loc[valid_scores].astype(int)
    df["Summary"] = df["Summary"].fillna("").astype(str)
    df["Text"] = df["Text"].fillna("").astype(str)
    df["clean_body"] = df["Text"].map(clean)
    nonempty = df["clean_body"].ne("")
    quality["empty_text_removed"] = int((~nonempty).sum())
    df = df.loc[nonempty].copy()
    df["label"] = df["Score"].apply(star_to_label)

    # Deduplicate before splitting: HTML/case variants otherwise leak between sets.
    exact_duplicates = int(df.duplicated(subset="Text").sum())
    duplicates = int(df.duplicated(subset="clean_body").sum())
    quality.update(
        exact_duplicates_removed=exact_duplicates,
        cleaning_equivalent_duplicates_removed=duplicates - exact_duplicates,
        duplicates_removed=duplicates,
        conflicting_duplicate_groups=int(
            df.groupby("clean_body")["label"].nunique().gt(1).sum()
        ),
    )
    df = df.drop_duplicates(subset="clean_body").reset_index(drop=True)
    df["doc"] = (df["Summary"] + " . " + df["Text"]).map(clean)
    df["n_words"] = df["clean_body"].str.split().str.len()
    df.attrs["data_quality"] = quality
    return df, n_raw


def split_data(df):
    """Use the same stratified outer split in training and follow-up scripts."""
    return train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=SEED
    )


def make_vectorizers():
    # No stop-word list: it would delete "not", "no", "never" etc.
    bow = CountVectorizer(ngram_range=(1, 1), min_df=2)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True
    )
    return bow, tfidf


def evaluate(name, feats, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    pred_s = time.perf_counter() - t0
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=LABELS, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        yte, pred, labels=LABELS, average="macro", zero_division=0
    )
    _, _, fw, _ = precision_recall_fscore_support(
        yte, pred, labels=LABELS, average="weighted", zero_division=0
    )
    row = dict(model=name, features=feats,
               accuracy=accuracy_score(yte, pred),
               macro_precision=pm, macro_recall=rm, macro_f1=fm,
               weighted_f1=fw,
               fit_seconds=fit_s, predict_ms_per_1k=pred_s * 1_000_000 / len(yte))
    for i, lab in enumerate(LABELS):
        row[f"{lab}_precision"], row[f"{lab}_recall"], row[f"{lab}_f1"] = p[i], r[i], f[i]
    cm = confusion_matrix(yte, pred, labels=LABELS)
    return row, pred, cm


def main():
    df, n_raw = load()
    tr, te = split_data(df)
    ytr, yte = tr["label"].values, te["label"].values

    bow, tfidf = make_vectorizers()
    Xtr_b = bow.fit_transform(tr["doc"])
    Xte_b = bow.transform(te["doc"])
    Xtr_t = tfidf.fit_transform(tr["doc"])
    Xte_t = tfidf.transform(te["doc"])

    # Fixed initial settings. tune.py evaluates candidates without using test data.
    models = [
        ("Majority class", "-", DummyClassifier(strategy="most_frequent"), Xtr_b, Xte_b),
        ("Multinomial NB", "BoW", MultinomialNB(alpha=0.5), Xtr_b, Xte_b),
        ("Complement NB", "TF-IDF", ComplementNB(alpha=0.3), Xtr_t, Xte_t),
        ("Logistic Regression", "TF-IDF", LogisticRegression(C=5, max_iter=3000,
                                class_weight="balanced", random_state=SEED), Xtr_t, Xte_t),
        ("Linear SVM", "TF-IDF", LinearSVC(C=0.25, class_weight="balanced",
                                           random_state=SEED), Xtr_t, Xte_t),
    ]
    rows, preds, cms, fitted = [], {}, {}, {}
    for name, feats, m, Xa, Xb in models:
        row, pred, cm = evaluate(name, feats, m, Xa, ytr, Xb, yte)
        rows.append(row)
        preds[name] = pred
        cms[name] = cm.tolist()
        fitted[name] = m
        print(f"{name:22s} acc={row['accuracy']:.3f} macroF1={row['macro_f1']:.3f} "
              f"fit={row['fit_seconds']:.1f}s")

    RES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "model_comparison.csv", index=False)

    summary = dict(
        n_raw=n_raw, n_dedup=len(df), n_train=len(tr), n_test=len(te),
        seed=SEED, test_fraction=0.20, labels=LABELS,
        data_quality=df.attrs["data_quality"],
        model_settings={
            "Linear SVM": {"C": 0.25, "class_weight": "balanced"},
            "Logistic Regression": {"C": 5, "class_weight": "balanced", "max_iter": 3000},
            "Multinomial NB": {"alpha": 0.5},
            "Complement NB": {"alpha": 0.3},
            "parameter_source": "fixed initial settings; tuning.csv is a separate training-only comparison",
        },
        feature_settings={
            "BoW": {"ngram_range": [1, 1], "min_df": 2},
            "TF-IDF": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.9,
                       "sublinear_tf": True, "norm": "l2"},
        },
        label_counts=df["label"].value_counts().to_dict(),
        test_label_counts=te["label"].value_counts().to_dict(),
        bow_vocab=len(bow.vocabulary_), tfidf_vocab=len(tfidf.vocabulary_),
        mean_words=float(df["n_words"].mean()), median_words=float(df["n_words"].median()),
        confusion=cms,
    )
    with (RES / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    out = te[["Id", "Score", "label", "n_words", "Summary", "Text"]].copy()
    for name, p in preds.items():
        out["pred_" + name.replace(" ", "_")] = p
    out.to_csv(RES / "test_predictions.csv", index=False)

    joblib.dump(dict(bow=bow, tfidf=tfidf, models=fitted, Xtr_t=Xtr_t, ytr=ytr,
                     Xte_t=Xte_t, train_docs=tr["doc"].values, train_labels=ytr,
                     train_ids=tr["Id"].values, test_ids=te["Id"].values,
                     seed=SEED, labels=LABELS), ARTIFACTS)


if __name__ == "__main__":
    main()
