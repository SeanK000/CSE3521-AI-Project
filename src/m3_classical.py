"""Compare M3 TF-IDF variants and refit the validation winner on outer training."""

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from distilbert_model import DEFAULT_DATA_DIR, load_splits, save_json
from m3_features import VARIANTS, ReviewFeatureTransformer
from pipeline import LABELS, RES, ROOT, SEED, evaluate


DEFAULT_REPORT_DIR = RES / "m3_classical"
DEFAULT_OUTPUT = ROOT / "models" / "m3_classical.joblib"
FEATURE_LABELS = {
    "combined": "Combined TF-IDF",
    "separate": "Separate summary/body TF-IDF",
    "separate_negation": "Separate TF-IDF + negation",
    "separate_negation_contrast": "Separate TF-IDF + negation and contrast",
}


def classical_models():
    """Use M2's classifier settings to isolate the effect of the features."""
    return {
        "Linear SVM": LinearSVC(C=0.25, class_weight="balanced", random_state=SEED),
        "Logistic Regression": LogisticRegression(
            C=5, max_iter=3000, class_weight="balanced", random_state=SEED,
        ),
    }


def compare_variants(inner_train, validation):
    rows = []
    for variant in VARIANTS:
        transformer = ReviewFeatureTransformer(variant)
        start = time.perf_counter()
        training_features = transformer.fit_transform(inner_train)
        feature_fit_seconds = time.perf_counter() - start
        validation_features = transformer.transform(validation)
        for name, model in classical_models().items():
            metrics, _, _ = evaluate(
                name, FEATURE_LABELS[variant], model,
                training_features, inner_train["label"],
                validation_features, validation["label"],
            )
            rows.append(dict(
                metrics, variant=variant, C=model.C,
                feature_fit_seconds=feature_fit_seconds,
                n_features=training_features.shape[1],
                n_train=len(inner_train), n_validation=len(validation),
            ))
            print(f"{variant} / {name}: validation macro-F1={metrics['macro_f1']:.4f}",
                  flush=True)

    # Keep the configured feature/model order if macro-F1 is exactly tied.
    ranking = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, kind="stable")
    ranking = ranking.reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking


def run_classical(splits, report_dir=DEFAULT_REPORT_DIR):
    """Return the refitted winner's app entry and its held-out test metrics."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    with threadpool_limits(limits=1, user_api="blas"):
        ranking = compare_variants(splits["inner_train"], splits["validation"])
        ranking.to_csv(report_dir / "validation_comparison.csv", index=False)
        best = ranking.iloc[0]

        training, test = splits["outer_train"], splits["test"]
        variant = best["variant"]
        feature_label = FEATURE_LABELS[variant]
        name = f"M3 {best['model']}"
        transformer = ReviewFeatureTransformer(variant)
        start = time.perf_counter()
        training_features = transformer.fit_transform(training)
        feature_fit_seconds = time.perf_counter() - start
        test_features = transformer.transform(test)
        model = classical_models()[best["model"]]
        metrics, predictions, confusion = evaluate(
            name, feature_label, model, training_features, training["label"],
            test_features, test["label"],
        )

    metrics.update(
        evaluation_split="held-out test", variant=variant, C=model.C,
        validation_macro_f1=float(best["macro_f1"]),
        n_train=len(training), n_test=len(test), n_features=training_features.shape[1],
        feature_fit_seconds=feature_fit_seconds, labels=LABELS,
        confusion_matrix=confusion.tolist(),
    )
    save_json(report_dir / "test_metrics.json", metrics)
    output = test[["Id", "label"]].copy()
    output["predicted"] = predictions
    output["correct"] = output["label"].eq(output["predicted"])
    output.to_csv(report_dir / "test_predictions.csv", index=False)
    print(f"{name} ({variant}): test accuracy={metrics['accuracy']:.4f}, "
          f"macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return {"transformer": transformer, "model": model, "feature_label": feature_label}, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    splits, source = load_splits(args.data_dir)
    print(f"Loaded {source}", flush=True)
    entry, metrics = run_classical(splits, args.report_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"entry": entry, "metrics": metrics}, args.output, compress=3)


if __name__ == "__main__":
    main()
