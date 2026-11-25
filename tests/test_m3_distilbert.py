"""Data/metric tests for the DistilBERT script; torch is not required."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import distilbert_model as model


class DataTests(unittest.TestCase):
    def rows(self):
        labels = ["negative", "neutral", "positive", "positive",
                  "negative", "neutral"]
        return pd.DataFrame({
            "Id": list(range(1, 7)),
            "Summary": ["s1", "s2", "s3", None, "s5", "s6"],
            "Text": ["b1", "b2", "b3", "b4", "b5", "b6"],
            "clean_body": ["b1", "b2", "b3", "b4", "b5", "b6"],
            "label": labels,
            "label_id": [model.LABEL_TO_ID[label] for label in labels],
        })

    def write_prepared(self, directory):
        rows = self.rows()
        rows.to_json(directory / "reviews.jsonl", orient="records",
                     lines=True, force_ascii=False)
        orders = {
            "outer_train": [4, 3, 2, 1],
            "test": [6, 5],
            "inner_train": [3, 2, 1],
            "validation": [4],
        }
        for name, ids in orders.items():
            frame = rows.set_index("Id").loc[ids, ["label", "label_id"]]
            frame.reset_index().to_csv(directory / f"{name}_labels.csv", index=False)

    def test_prepared_splits_keep_file_order_and_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_prepared(directory)
            splits, source = model.load_splits(directory)
            self.assertIn("prepared", source)
            self.assertEqual(splits["outer_train"]["Id"].tolist(), [4, 3, 2, 1])
            self.assertEqual(splits["test"]["Id"].tolist(), [6, 5])
            self.assertEqual(splits["inner_train"]["Id"].tolist(), [3, 2, 1])
            self.assertEqual(splits["validation"]["Id"].tolist(), [4])

    def test_incomplete_prepared_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.rows().to_json(directory / "reviews.jsonl", orient="records",
                                lines=True, force_ascii=False)
            with self.assertRaises(ValueError):
                model.load_splits(directory)

    def test_overlapping_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_prepared(directory)
            splits, _ = model.load_splits(directory)
            splits["validation"] = splits["inner_train"].head(1).copy()
            with self.assertRaises(ValueError):
                model.verify_splits(splits)

    def test_text_pairs_keeps_summary_and_body_separate(self):
        frame = pd.DataFrame({"Summary": ["<b>Great</b>", None],
                              "Text": ["I liked it<br />", "It was fine"]})
        summaries, bodies = model.text_pairs(frame)
        self.assertEqual(summaries, ["great", ""])
        self.assertEqual(bodies, ["i liked it", "it was fine"])


class MetricTests(unittest.TestCase):
    def test_balanced_class_weights(self):
        weights = model.compute_class_weights([0, 0, 1, 2])
        np.testing.assert_allclose(weights, [4 / 6, 4 / 3, 4 / 3])
        self.assertIsNone(model.compute_class_weights([0, 1], enabled=False))

    def test_missing_class_gets_zero_weight(self):
        weights = model.compute_class_weights([0, 0, 2, 2])
        np.testing.assert_allclose(weights, [4 / 6, 0.0, 4 / 6])

    def test_metrics_match_label_order(self):
        metrics = model.classification_metrics(
            [0, 1, 2, 2], [0, 2, 2, 2], loss=0.5,
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["per_class"]["negative"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["positive"]["recall"], 1.0)
        self.assertEqual(metrics["per_class"]["neutral"]["support"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 0, 1], [0, 0, 2]])
        self.assertEqual(metrics["loss"], 0.5)

    def test_prediction_frame_has_probability_columns(self):
        frame = pd.DataFrame({"Id": [1, 2], "label": ["negative", "positive"]})
        probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        output = model.prediction_frame(frame, probabilities, [0, 2])
        self.assertEqual(output["predicted"].tolist(), ["negative", "positive"])
        self.assertTrue(output["correct"].all())
        self.assertEqual(
            list(output.columns),
            ["Id", "label", "predicted", "prob_negative", "prob_neutral",
             "prob_positive", "correct"],
        )
        with self.assertRaises(ValueError):
            model.prediction_frame(frame, probabilities[:1], [0, 2])


if __name__ == "__main__":
    unittest.main()
