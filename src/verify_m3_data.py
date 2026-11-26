"""Verify persisted M3 data/features, without fitting or scoring classifiers."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from prepare_m3_data import ROOT, DEFAULT_OUTPUT, load_split, file_hash, text_hash, LABEL_TO_ID
from m3_features import VARIANTS


def verify(output, report_dir):
    output, report_dir = Path(output), Path(report_dir)
    contract = json.loads((output / "artifact_manifest.json").read_text())
    audit = json.loads((report_dir / "data_audit.json").read_text())
    for name, expected in contract["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    for name, expected in audit["source_files"].items():
        assert text_hash(ROOT / name) == expected, f"M2 baseline changed: {name}"

    split = {name: load_split(output, name) for name in ["outer_train", "test", "inner_train", "validation"]}
    for name, frame in split.items():
        assert len(frame) == audit["splits"][name]["rows"]
        assert frame.Id.is_unique and frame.clean_body.notna().all()
        np.testing.assert_array_equal(frame.label.map(LABEL_TO_ID), frame.label_id)
    outer, test, inner, validation = [split[name] for name in split]
    assert set(outer.Id).isdisjoint(test.Id)
    assert set(outer.clean_body).isdisjoint(test.clean_body)
    assert set(inner.Id).isdisjoint(validation.Id)
    assert set(inner.Id) | set(validation.Id) == set(outer.Id)
    predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
    np.testing.assert_array_equal(predictions.Id, test.Id)
    np.testing.assert_array_equal(predictions.label, test.label)

    manifest = pd.read_csv(output / "split_manifest.csv")
    assert len(manifest) == len(outer) + len(test)
    assert manifest.Id.is_unique
    assert set(manifest.loc[manifest.cv_fold >= 0, "Id"]) == set(outer.Id)
    assert set(manifest.loc[manifest.cv_fold == -1, "Id"]) == set(test.Id)
    assert set(manifest.loc[manifest.outer_split == "outer_train", "cv_fold"]) == {0, 1, 2}
    for fold in range(3):
        held = manifest[manifest.cv_fold == fold]
        fitted = manifest[(manifest.cv_fold >= 0) & (manifest.cv_fold != fold)]
        assert set(held.Id).isdisjoint(fitted.Id)
        assert set(held.body_sha256).isdisjoint(fitted.body_sha256)
        assert set(held.label) == set(LABEL_TO_ID)

    verified = {}
    for variant in VARIANTS:
        directory = output / "features" / variant
        transformer = joblib.load(directory / "transformer.joblib")
        assert transformer.n_fit_rows_ == len(inner)
        assert transformer.n_output_features_ == len(transformer.get_feature_names_out())
        for role, frame in [("inner_train", inner), ("validation", validation)]:
            matrix = sparse.load_npz(directory / f"{role}.npz")
            regenerated = transformer.transform(frame)
            assert matrix.shape == (len(frame), transformer.n_output_features_)
            delta = matrix - regenerated
            assert delta.nnz == 0 or abs(delta.data).max() < 1e-12
            assert np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
        verified[variant] = "All rows match reloaded transformer and ordered labels"
        print(f"Verified {variant}", flush=True)

    m2_variants = pd.read_csv(ROOT / "results/experiments.csv")
    for row, variant in zip(m2_variants.itertuples(), VARIANTS[:3]):
        assert int(row.n_features) == audit["candidate_features"][variant]["inner_train_shape"][1]
    result = {
        "checksums_verified": len(contract["files"]), "m2_baseline_files_unchanged": True,
        "m2_test_ids_labels_and_order_match": True, "train_test_id_and_clean_body_overlap": 0,
        "inner_train_validation_id_overlap": 0, "cv_folds_train_only": True,
        "first_three_vocab_sizes_match_m2_experiments": True,
        "matrix_verification": verified, "model_training_performed": False,
        "held_out_test_scoring_performed": False,
    }
    (report_dir / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/m3_data")
    args = parser.parse_args()
    verify(args.output_dir, args.report_dir)
