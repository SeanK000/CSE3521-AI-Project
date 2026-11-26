"""Train and compare the M3 models, then build the Streamlit model bundle."""

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from distilbert_model import load_splits, save_json
from m3_classical import run_classical
from m3_neural import run_neural
from pipeline import DATA, FIG, LABELS, RES, ROOT
from prepare_m3_data import SAMPLE_SHA256, text_hash


BUNDLE = ROOT / "models" / "m3_models.joblib"


def load_training_data():
    if DATA.is_file() and text_hash(DATA) != SAMPLE_SHA256:
        raise ValueError("The shared CSV differs from the locked seed-47 sample.")
    splits, source = load_splits()
    reference = pd.read_csv(RES / "test_predictions.csv", usecols=["Id", "label"])
    test = splits["test"][["Id", "label"]].reset_index(drop=True)
    if not test.equals(reference):
        raise ValueError("M3 test review IDs or labels differ from the M2 baseline.")
    return splits, source


def write_outputs(results, splits, source):
    """Publish M2 and M3 scores together and save the matched inference models."""
    baselines = pd.read_csv(RES / "model_comparison.csv")
    columns = ["model", "features", "accuracy", "macro_precision", "macro_recall",
               "macro_f1", "weighted_f1", "fit_seconds", "predict_ms_per_1k"]
    comparison = baselines[columns].assign(milestone="M2")
    new_rows = pd.DataFrame([{key: metrics[key] for key in columns} for _, metrics in results])
    comparison = pd.concat([comparison, new_rows.assign(milestone="M3")], ignore_index=True)
    output_dir = RES / "m3"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    with (RES / "summary.json").open(encoding="utf-8") as stream:
        matrices = json.load(stream)["confusion"]
    matrices.update({metrics["model"]: metrics["confusion_matrix"] for _, metrics in results})
    save_json(output_dir / "confusion_matrices.json", {"labels": LABELS, "models": matrices})

    primary = max((metrics for _, metrics in results), key=lambda row: row["validation_macro_f1"])
    bundle = {
        "models": {metrics["model"]: entry for entry, metrics in results},
        "primary_model": primary["model"],
    }
    save_json(output_dir / "run_metadata.json", {
        "split_source": source.split(" (", 1)[0],
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "sample_sha256_lf": SAMPLE_SHA256,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn")},
        "primary_model": primary["model"],
        "primary_selection": "highest training-only validation macro F1 among M3 models",
        "timing": "classifier fitting and prediction only, excluding preprocessing",
    })
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=BUNDLE.parent, prefix=".m3-models-", suffix=".joblib",
                            delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, BUNDLE)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(comparison[["model", "accuracy", "macro_f1", "fit_seconds"]].to_string(index=False))
    print(f"Saved app bundle to {BUNDLE}", flush=True)


def main():
    splits, source = load_training_data()
    print(f"Loaded {source}", flush=True)
    classical = run_classical(splits, RES / "m3_classical")
    with threadpool_limits(limits=2):
        neural = run_neural(splits, RES / "m3_neural", FIG / "m3_neural_loss.png")
    write_outputs([classical, neural], splits, source)


if __name__ == "__main__":
    main()
