"""Freeze M2-compatible data, prepare M3 feature candidates, and audit outputs.

Run from the repository root: python src/prepare_m3_data.py
No classifier is fitted, no test predictions are made, and no M2 results change.
"""

import argparse
import hashlib
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

import pipeline
from experiments import validation_split, NEGATION, TOKENS
from m3_features import ReviewFeatureTransformer, VARIANTS, contrast_parts, feature_blocks


BASE_COMMIT = "ee71c7e2d0d23edf4d92f9f036d1568d747ec68a"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "m3"
SAMPLE_SHA256 = "57729385c4f7c892fbd8abfdb496897d24fedcfdaa90196f15345bca08e9cb20"
LABEL_TO_ID = {label: index for index, label in enumerate(pipeline.LABELS)}
BASELINE_FILES = ["src/pipeline.py", "src/experiments.py", "results/summary.json", "results/test_predictions.csv"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def body_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def counts(frame):
    return {label: int((frame["label"] == label).sum()) for label in pipeline.LABELS}


def load_split(output_dir, split):
    """Load raw and cleaned text in exactly the order of a prepared matrix.

    split: outer_train, test, inner_train, validation. Label IDs are metadata,
    never transformer inputs. Deep-model owners can tokenize Summary/Text raw.
    """
    output_dir = Path(output_dir)
    rows = pd.read_json(output_dir / "reviews.jsonl", lines=True).set_index("Id", drop=False)
    ordering = pd.read_csv(output_dir / f"{split}_labels.csv")
    return rows.loc[ordering["Id"].tolist()].reset_index(drop=True)


def make_manifests(clean, outer_train, test, inner_train, validation):
    manifest = clean[["Id", "label"]].copy()
    manifest["body_sha256"] = clean["clean_body"].map(body_hash)
    manifest["label_id"] = clean["label"].map(LABEL_TO_ID)
    manifest["outer_split"] = np.where(manifest.Id.isin(test.Id), "test", "outer_train")
    manifest["inner_split"] = manifest.Id.map({
        **{int(i): "inner_train" for i in inner_train.Id},
        **{int(i): "validation" for i in validation.Id},
        **{int(i): "not_used" for i in test.Id},
    })
    # Public training-only folds. -1 marks held-out test rows, never a CV fold.
    fold_by_id = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=pipeline.SEED)
    for fold, (_, validation_indices) in enumerate(cv.split(outer_train, outer_train.label)):
        for review_id in outer_train.iloc[validation_indices].Id:
            fold_by_id[int(review_id)] = fold
    manifest["cv_fold"] = manifest.Id.map(fold_by_id).fillna(-1).astype(int)
    return manifest.reset_index(drop=True)


def duplicate_audit(raw):
    """Explain M2's existing keep-first policy without changing any labels."""
    frame = raw.copy()
    frame["clean_body"] = frame.Text.fillna("").map(pipeline.clean)
    frame["label"] = frame.Score.map(pipeline.star_to_label)
    first = frame.drop_duplicates("clean_body").set_index("clean_body")
    duplicates = frame.loc[frame.duplicated("clean_body")].copy()
    duplicates["kept_id"] = duplicates.clean_body.map(first.Id)
    duplicates["kept_label"] = duplicates.clean_body.map(first.label)
    duplicates["label_conflict"] = duplicates.label != duplicates.kept_label
    duplicates["body_sha256"] = duplicates.clean_body.map(body_hash)
    return duplicates[["Id", "label", "kept_id", "kept_label", "label_conflict", "body_sha256"]]


def verify_m2(clean, outer_train, test, n_raw):
    summary = json.loads((ROOT / "results/summary.json").read_text())
    predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
    actual = {"n_raw": n_raw, "n_dedup": len(clean), "n_train": len(outer_train), "n_test": len(test)}
    if any(actual[key] != summary[key] for key in actual):
        raise ValueError(f"M2 split counts differ: {actual}. Stop rather than silently replace the shared split.")
    if test.Id.tolist() != predictions.Id.tolist() or test.label.tolist() != predictions.label.tolist():
        raise ValueError("Test IDs/labels/order differ from the submitted M2 predictions.")
    if counts(clean) != summary["label_counts"]:
        raise ValueError("Cleaned class counts differ from M2.")
    if set(outer_train.Id) & set(test.Id) or set(outer_train.clean_body) & set(test.clean_body):
        raise ValueError("Train/test leakage detected.")
    return summary


def prepare(sample, output_dir, report_dir):
    sample, output_dir, report_dir = Path(sample), Path(output_dir), Path(report_dir)
    if file_hash(sample) != SAMPLE_SHA256:
        raise ValueError("Input is not the locked seed-47 team sample; use the unchanged original CSV.")
    original_hashes = {name: file_hash(ROOT / name) for name in BASELINE_FILES}
    old_data = pipeline.DATA
    try:
        pipeline.DATA = sample
        clean, n_raw = pipeline.load()
    finally:
        pipeline.DATA = old_data
    if clean.Id.duplicated().any():
        raise ValueError("Review IDs are not unique.")
    outer_train, test = pipeline.split_data(clean)
    inner_train, validation = validation_split(clean)
    m2_summary = verify_m2(clean, outer_train, test, n_raw)
    assert set(inner_train.Id).isdisjoint(validation.Id)
    assert set(inner_train.Id) | set(validation.Id) == set(outer_train.Id)
    assert set(inner_train.Id).isdisjoint(test.Id) and set(validation.Id).isdisjoint(test.Id)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Verified M2 data and exact held-out test IDs/order.", flush=True)
    manifest = make_manifests(clean, outer_train, test, inner_train, validation)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    data = clean[["Id", "Summary", "Text", "clean_body", "doc", "label", "n_words"]].copy()
    data["label_id"] = data.label.map(LABEL_TO_ID)
    data.to_json(output_dir / "reviews.jsonl", orient="records", lines=True, force_ascii=False)
    split_frames = {"outer_train": outer_train, "test": test, "inner_train": inner_train, "validation": validation}
    for name, frame in split_frames.items():
        labels = frame[["Id", "label"]].copy()
        labels["label_id"] = labels.label.map(LABEL_TO_ID)
        labels.to_csv(output_dir / f"{name}_labels.csv", index=False)
    duplicates = duplicate_audit(pd.read_csv(sample))
    duplicates.to_csv(output_dir / "duplicate_audit.csv", index=False)

    # Structural compatibility only. No M2 classifier retraining or test scoring.
    bow, tfidf = pipeline.make_vectorizers()
    bow.fit(outer_train.doc)
    tfidf.fit(outer_train.doc)
    dimensions = {"bow": len(bow.vocabulary_), "tfidf": len(tfidf.vocabulary_)}
    assert dimensions == {"bow": m2_summary["bow_vocab"], "tfidf": m2_summary["tfidf_vocab"]}
    del bow, tfidf

    # Preserve feature fit boundaries from M2 experiments (inner_train only).
    candidates = {}
    for variant in VARIANTS:
        print(f"Preparing {variant} on inner training only...", flush=True)
        transformer = ReviewFeatureTransformer(variant)
        x_train = transformer.fit_transform(inner_train)
        x_validation = transformer.transform(validation)
        destination = output_dir / "features" / variant
        destination.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(destination / "inner_train.npz", x_train)
        sparse.save_npz(destination / "validation.npz", x_validation)
        joblib.dump(transformer, destination / "transformer.joblib")
        candidates[variant] = {
            "fit_split": "inner_train", "fit_rows": len(inner_train),
            "inner_train_shape": list(x_train.shape), "validation_shape": list(x_validation.shape),
            "block_sizes": transformer.block_sizes_,
            "inner_train_nonzero": int(x_train.nnz), "validation_nonzero": int(x_validation.nnz),
            "zero_inner_train_rows": int((x_train.getnnz(axis=1) == 0).sum()),
            "zero_validation_rows": int((x_validation.getnnz(axis=1) == 0).sum()),
        }
        assert x_train.shape[1] == x_validation.shape[1]
        assert np.isfinite(x_train.data).all() and np.isfinite(x_validation.data).all()
        del transformer, x_train, x_validation

    coverage = {}
    has_negation = inner_train.clean_body.map(lambda doc: any(NEGATION.fullmatch(token) for token in TOKENS.findall(doc)))
    has_contrast = inner_train.Text.map(lambda doc: any(contrast_parts(doc)))
    for label in pipeline.LABELS:
        mask = inner_train.label == label
        coverage[label] = {"rows": int(mask.sum()), "negation_marker_rows": int((has_negation & mask).sum()),
                           "contrast_rule_rows": int((has_contrast & mask).sum())}
    # Examples come from inner training only, selected by ID, not by model error.
    examples = inner_train.loc[has_contrast].sort_values("Id").groupby("label", sort=False).head(2)
    example_rows = []
    for row in examples.itertuples():
        before, after = contrast_parts(row.Text)
        example_rows.append({"Id": int(row.Id), "label": row.label,
                             "before": before[:240], "after": after[:240]})

    role_counts = {name: {"rows": len(frame), "labels": counts(frame)} for name, frame in split_frames.items()}
    software = {name: version(name) for name in ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]}
    audit = {
        "base_commit": BASE_COMMIT, "sample_sha256": file_hash(sample),
        "source_files": original_hashes, "n_raw": n_raw, "n_retained": len(clean),
        "seed": pipeline.SEED, "label_to_id": LABEL_TO_ID,
        "m2_data_quality": clean.attrs["data_quality"], "splits": role_counts,
        "m2_baseline_dimensions": dimensions, "m2_test_ids_and_order_match": True,
        "outer_id_overlap": 0, "outer_clean_body_overlap": 0,
        "inner_id_overlap": 0, "cv_folds": 3, "test_rows_in_cv": 0,
        "candidate_features": candidates, "inner_training_rule_coverage": coverage,
        "training_contrast_examples": example_rows, "software": software,
        "python": platform.python_version(), "model_training_performed": False,
        "test_predictions_performed": False,
        "pending": ["Model trainers: training-only feature comparisons and selection",
                    "ML engineer: deep-model/tokenizer choice and maximum sequence length",
                    "App engineer: integrate the selected model with its matching transformer"],
    }
    # Verify this data-only run left all baseline evidence untouched.
    assert original_hashes == {name: file_hash(ROOT / name) for name in BASELINE_FILES}
    (report_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    contract = {"base_commit": BASE_COMMIT, "sample_sha256": file_hash(sample),
                "label_to_id": LABEL_TO_ID, "software": software,
                "files": {str(path.relative_to(output_dir)): file_hash(path)
                          for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"}}
    (output_dir / "artifact_manifest.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps({"splits": role_counts, "candidates": candidates}, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=pipeline.DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results" / "m3_data")
    args = parser.parse_args()
    prepare(args.sample, args.output_dir, args.report_dir)


if __name__ == "__main__":
    main()
