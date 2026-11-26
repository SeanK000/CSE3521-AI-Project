"""Train the M3 feedforward neural network and save its evaluation."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from distilbert_model import classification_metrics, prediction_frame, save_json
from m3_features import ReviewFeatureTransformer
from pipeline import LABELS, SEED


NAME = "M3 Feedforward Neural Network"
FEATURE_LABEL = "TF-IDF with chi-square selection"
HIDDEN_LAYERS = (64, 32)
FEATURE_PERCENTILE = 6
BATCH_SIZE = 512
MAX_EPOCHS = 12
PATIENCE = 3


def float32_features(matrix):
    return matrix.astype(np.float32, copy=False)


def make_transformer():
    return make_pipeline(
        ReviewFeatureTransformer("combined"),
        SelectPercentile(chi2, percentile=FEATURE_PERCENTILE),
        FunctionTransformer(float32_features),
    )


def make_model():
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        batch_size=BATCH_SIZE,
        random_state=np.random.RandomState(SEED),
    )


def save_loss_curves(history, best_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], "o-", label="Validation")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend(frameon=False)
    axes[1].plot(history["epoch"], history["validation_macro_f1"], "o-", color="#18806e")
    axes[1].set_ylabel("Validation macro F1")
    for axis in axes:
        axis.axvline(best_epoch, color="#52606d", linestyle="--", linewidth=1)
        axis.set_xlabel("Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Feedforward network: training and validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_neural(splits, report_dir, figure_path):
    """Select an epoch on validation, then refit on outer train and score test."""
    inner, validation, outer, test = (
        splits[name] for name in ("inner_train", "validation", "outer_train", "test")
    )
    selection_start = time.perf_counter()
    transformer = make_transformer()
    X_inner = transformer.fit_transform(inner, inner["label"])
    X_val = transformer.transform(validation)
    y_inner, y_val = inner["label"].to_numpy(), validation["label"].to_numpy()
    weights = compute_sample_weight("balanced", y_inner)
    labels = np.array(LABELS)

    model = make_model()
    history = []
    best_epoch, best_f1 = 0, -1.0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(X_inner, y_inner, classes=labels, sample_weight=weights)
        train_probabilities = model.predict_proba(X_inner)
        validation_probabilities = model.predict_proba(X_val)
        predictions = model.classes_[validation_probabilities.argmax(axis=1)]
        score = f1_score(y_val, predictions, labels=LABELS, average="macro", zero_division=0)
        row = {
            "epoch": epoch,
            "train_loss": log_loss(y_inner, train_probabilities, labels=model.classes_),
            "validation_loss": log_loss(y_val, validation_probabilities, labels=model.classes_),
            "validation_macro_f1": score,
        }
        history.append(row)
        print(f"MLP epoch {epoch:2d}: train loss={row['train_loss']:.4f}, "
              f"validation loss={row['validation_loss']:.4f}, macro F1={score:.4f}", flush=True)
        if score > best_f1:
            best_epoch, best_f1 = epoch, score
        elif epoch - best_epoch >= PATIENCE:
            break
    selection_seconds = time.perf_counter() - selection_start
    del model, transformer, X_inner, X_val

    transformer = make_transformer()
    start = time.perf_counter()
    X_outer = transformer.fit_transform(outer, outer["label"])
    feature_fit_seconds = time.perf_counter() - start
    X_test = transformer.transform(test)
    y_outer = outer["label"].to_numpy()
    weights = compute_sample_weight("balanced", y_outer)
    model = make_model()
    print(f"Refitting MLP for {best_epoch} epochs on the {len(outer):,} training reviews.", flush=True)
    start = time.perf_counter()
    for _ in range(best_epoch):
        model.partial_fit(X_outer, y_outer, classes=labels, sample_weight=weights)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = model.predict_proba(X_test)
    predict_ms_per_1k = (time.perf_counter() - start) * 1_000_000 / len(test)

    ids = {label: index for index, label in enumerate(LABELS)}
    predicted_ids = probabilities.argmax(axis=1)
    metrics = classification_metrics(
        test["label"].map(ids), predicted_ids,
        log_loss(test["label"], probabilities, labels=model.classes_),
    )
    metrics.update(
        model=NAME, features=FEATURE_LABEL, fit_seconds=fit_seconds,
        feature_fit_seconds=feature_fit_seconds, selection_seconds=selection_seconds,
        predict_ms_per_1k=predict_ms_per_1k, best_epoch=best_epoch,
        validation_macro_f1=best_f1, hidden_layers=HIDDEN_LAYERS,
        feature_percentile=FEATURE_PERCENTILE, features_retained=X_outer.shape[1],
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
        macro_precision=float(np.mean([row["precision"] for row in metrics["per_class"].values()])),
        macro_recall=float(np.mean([row["recall"] for row in metrics["per_class"].values()])),
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(history)
    history.to_csv(report_dir / "history.csv", index=False)
    save_loss_curves(history, best_epoch, Path(figure_path))
    save_json(report_dir / "test_metrics.json", metrics)
    prediction_frame(test, probabilities, predicted_ids).to_csv(
        report_dir / "test_predictions.csv", index=False,
    )
    entry = {"transformer": transformer, "model": model, "feature_label": FEATURE_LABEL}
    return entry, metrics
