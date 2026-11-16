"""CSE 3521 Milestone 2 - feature pipeline, classical models, and evaluation.

Run from the repo root:
    python src/pipeline.py
Outputs go to results/ (JSON + CSV) and figures/ (PNG).
"""
import html
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
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
DATA = ROOT / "group_sample_seed47.csv"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
LABELS = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------- data
def star_to_label(s):
    return "negative" if s <= 2 else ("neutral" if s == 3 else "positive")


def clean(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)          # <br /> tags etc.
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load():
    df = pd.read_csv(DATA)
    n_raw = len(df)
    df["Summary"] = df["Summary"].fillna("")
    # Same review text is posted under several product variants -> drop
    # exact duplicates BEFORE splitting so no review is in both train and test.
    df = df.drop_duplicates(subset="Text").reset_index(drop=True)
    df["label"] = df["Score"].apply(star_to_label)
    # Summary is a strong, short sentiment signal; prepend it to the body.
    df["doc"] = (df["Summary"] + " . " + df["Text"]).map(clean)
    df["n_words"] = df["Text"].map(clean).str.split().str.len()
    return df, n_raw


# ---------------------------------------------------------------- features
def make_vectorizers():
    # No stop-word list: it would delete "not", "no", "never" etc.
    bow = CountVectorizer(ngram_range=(1, 1), min_df=2)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                            sublinear_tf=True)
    return bow, tfidf


# ---------------------------------------------------------------- eval
def evaluate(name, feats, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); pred_s = time.perf_counter() - t0
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=LABELS, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
    row = dict(model=name, features=feats,
               accuracy=accuracy_score(yte, pred),
               macro_precision=pm, macro_recall=rm, macro_f1=fm,
               weighted_f1=fw,
               fit_seconds=fit_s, predict_ms_per_1k=1000 * pred_s / len(yte) * 1000)
    for i, lab in enumerate(LABELS):
        row[f"{lab}_precision"], row[f"{lab}_recall"], row[f"{lab}_f1"] = p[i], r[i], f[i]
    cm = confusion_matrix(yte, pred, labels=LABELS)
    return row, pred, cm


def main():
    df, n_raw = load()
    tr, te = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
    ytr, yte = tr["label"].values, te["label"].values

    bow, tfidf = make_vectorizers()
    Xtr_b = bow.fit_transform(tr["doc"]); Xte_b = bow.transform(te["doc"])
    Xtr_t = tfidf.fit_transform(tr["doc"]); Xte_t = tfidf.transform(te["doc"])

    # C values chosen by 3-fold CV on the training split only (src/tune.py).
    models = [
        ("Majority class", "-", DummyClassifier(strategy="most_frequent"), Xtr_b, Xte_b),
        ("Multinomial NB", "BoW", MultinomialNB(alpha=0.5), Xtr_b, Xte_b),
        ("Complement NB", "TF-IDF", ComplementNB(alpha=0.3), Xtr_t, Xte_t),
        ("Logistic Regression", "TF-IDF", LogisticRegression(C=5, max_iter=3000,
                                class_weight="balanced"), Xtr_t, Xte_t),
        ("Linear SVM", "TF-IDF", LinearSVC(C=0.25, class_weight="balanced"), Xtr_t, Xte_t),
    ]
    rows, preds, cms, fitted = [], {}, {}, {}
    for name, feats, m, Xa, Xb in models:
        row, pred, cm = evaluate(name, feats, m, Xa, ytr, Xb, yte)
        rows.append(row); preds[name] = pred; cms[name] = cm.tolist(); fitted[name] = m
        print(f"{name:22s} acc={row['accuracy']:.3f} macroF1={row['macro_f1']:.3f} "
              f"fit={row['fit_seconds']:.1f}s")

    res = pd.DataFrame(rows)
    res.to_csv(RES / "model_comparison.csv", index=False)

    summary = dict(
        n_raw=n_raw, n_dedup=len(df), n_train=len(tr), n_test=len(te),
        label_counts=df["label"].value_counts().to_dict(),
        test_label_counts=te["label"].value_counts().to_dict(),
        bow_vocab=len(bow.vocabulary_), tfidf_vocab=len(tfidf.vocabulary_),
        mean_words=float(df["n_words"].mean()), median_words=float(df["n_words"].median()),
        confusion=cms,
    )
    json.dump(summary, open(RES / "summary.json", "w"), indent=2)

    out = te[["Id", "Score", "label", "n_words", "Summary", "Text"]].copy()
    for name, p in preds.items():
        out["pred_" + name.replace(" ", "_")] = p
    out.to_csv(RES / "test_predictions.csv", index=False)

    joblib.dump(dict(bow=bow, tfidf=tfidf, models=fitted, Xtr_t=Xtr_t, ytr=ytr,
                     Xte_t=Xte_t, train_docs=tr["doc"].values, train_labels=ytr),
                RES / "artifacts.joblib")


if __name__ == "__main__":
    main()
