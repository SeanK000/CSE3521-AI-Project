"""Compare optional SVM features on validation data from the training split.

Run with ``python src/experiments.py``. The outer 20% test set is not used to
choose a variant. Results are written to results/experiments.csv.
"""

import re

import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from pipeline import LABELS, RES, SEED, clean, load, make_vectorizers, split_data

TOKENS = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")
NEGATION = re.compile(r"(?:not|no|never|cannot|\w+n['’]t)", re.IGNORECASE)


def neg_scope(text):
    """Mark words after a negation token until punctuation ends the scope.

    This is a feature experiment, not a grammatical parser. Contractions retain
    their apostrophes, including curly apostrophes, so ``can't`` is recognized.
    """
    output = []
    in_scope = False
    for token in TOKENS.findall(text):
        if not token[0].isalnum() and token[0] != "_":
            output.append(token)
            in_scope = False
        elif NEGATION.fullmatch(token):
            output.append(token)
            in_scope = True
        else:
            output.append(f"{token}_NEG" if in_scope else token)
    return " ".join(output)


def clean_with_negation(text):
    return neg_scope(clean(text))


def validation_split(data):
    """Reserve validation rows inside the shared outer training split."""
    outer_training, _ = split_data(data)
    return train_test_split(
        outer_training,
        test_size=0.2,
        stratify=outer_training["label"],
        random_state=SEED,
    )


def evaluate_variant(name, training_features, validation_features, training, validation):
    model = LinearSVC(C=0.25, class_weight="balanced", random_state=SEED)
    model.fit(training_features, training["label"])
    predictions = model.predict(validation_features)
    result = {
        "variant": name,
        "evaluation_split": "training-only validation",
        "n_train": len(training),
        "n_validation": len(validation),
        "accuracy": accuracy_score(validation["label"], predictions),
        "macro_f1": f1_score(
            validation["label"], predictions, labels=LABELS,
            average="macro", zero_division=0,
        ),
        "neutral_f1": f1_score(
            validation["label"], predictions, labels=["neutral"],
            average="macro", zero_division=0,
        ),
        "n_features": training_features.shape[1],
    }
    print(result)
    return result


def run_experiments(training, validation):
    _, vectorizer = make_vectorizers()
    results = [evaluate_variant(
        "baseline: summary+text in one TF-IDF",
        vectorizer.fit_transform(training["doc"]),
        vectorizer.transform(validation["doc"]),
        training, validation,
    )]

    variants = [
        ("separate summary & body TF-IDF", clean),
        ("separate + negation scoping", clean_with_negation),
    ]
    for name, preprocess in variants:
        _, summary_vectorizer = make_vectorizers()
        _, body_vectorizer = make_vectorizers()
        training_features = hstack([
            summary_vectorizer.fit_transform(training["Summary"].map(preprocess)),
            body_vectorizer.fit_transform(training["Text"].map(preprocess)),
        ]).tocsr()
        validation_features = hstack([
            summary_vectorizer.transform(validation["Summary"].map(preprocess)),
            body_vectorizer.transform(validation["Text"].map(preprocess)),
        ]).tocsr()
        results.append(evaluate_variant(
            name, training_features, validation_features, training, validation,
        ))
    return pd.DataFrame(results)


def main():
    data, _ = load()
    training, validation = validation_split(data)
    results = run_experiments(training, validation)
    RES.mkdir(exist_ok=True)
    results.to_csv(RES / "experiments.csv", index=False)


if __name__ == "__main__":
    main()
