"""Data-only tests: python -m unittest discover -s tests -p 'test_m3*.py' -v."""
import io
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import clean
from experiments import neg_scope
from m3_features import ReviewFeatureTransformer, contrast_parts, feature_blocks, VARIANTS


class FeatureTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "Summary": ["great tea", "bad snack", None, "mixed review", "new coffee", "tasty food"],
            "Text": ["I love tea but it costs too much.", "I can't eat this, it tastes bad.",
                     "Fresh food and nice packaging.", "Although it smells good, it tastes bitter.",
                     "Not bad. I will buy it again.", "Nothing but good ingredients here."],
        })

    def estimator(self, variant):
        return ReviewFeatureTransformer(variant, min_df=1, max_df=1.0)

    def test_contrast_has_direction_not_sentiment(self):
        self.assertEqual(contrast_parts("Great flavor, but too expensive."), ("great flavor,", "too expensive."))
        self.assertEqual(contrast_parts("Although tasty, it is overpriced."), ("tasty", "it is overpriced."))
        self.assertEqual(contrast_parts("Good but pricey; however I would buy again."),
                         ("good but pricey;", "i would buy again."))

    def test_contrast_ignores_idioms_and_incomplete_clauses(self):
        for text in ["Nothing but good food.", "Not only tasty but also cheap.",
                     "Butter is tasty.", "Good though.", "Although tasty", "Great food."]:
            with self.subTest(text=text):
                self.assertEqual(contrast_parts(text), ("", ""))

    def test_negation_reuses_m2_punctuation_contract(self):
        self.assertEqual(neg_scope(clean("I can’t recommend it. It is fresh.")),
                         "i can’t recommend_NEG it_NEG . it is fresh .")
        self.assertEqual(neg_scope("not good, fresh"), "not good_NEG , fresh")
        # Full-body scope still follows M2; separate contrast channels reset it.
        frame = pd.DataFrame({"Text": ["not tasty but affordable"]})
        blocks = feature_blocks(frame, "separate_negation_contrast")
        self.assertEqual(blocks["contrast_before"], ["not tasty_NEG"])
        self.assertEqual(blocks["contrast_after"], ["affordable"])

    def test_combined_matches_existing_m2_documents(self):
        frame = self.frame()
        expected = (frame.Summary.fillna("") + " . " + frame.Text).map(clean).tolist()
        self.assertEqual(feature_blocks(frame, "combined")["combined"], expected)

    def test_first_three_variants_match_m2_vectorization(self):
        frame = self.frame()
        options = dict(ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True)
        for variant in VARIANTS[:3]:
            blocks = feature_blocks(frame, variant)
            reference = sparse.hstack([TfidfVectorizer(**options).fit_transform(docs)
                                       for docs in blocks.values()], format="csr")
            actual = self.estimator(variant).fit_transform(frame)
            np.testing.assert_allclose(actual.toarray(), reference.toarray())

    def test_test_only_words_and_labels_are_not_features(self):
        frame = self.frame()
        frame["Score"] = [5, 1, 3, 3, 5, 4]
        frame["label"] = "targetmetadata"
        held_out = pd.DataFrame({"Text": ["validationuniqueword but validationotherword"],
                                 "Summary": ["unseensummarytoken"]})
        for variant in VARIANTS:
            estimator = self.estimator(variant).fit(frame)
            before = estimator.get_feature_names_out().tolist()
            result = estimator.transform(held_out)
            self.assertEqual(before, estimator.get_feature_names_out().tolist())
            self.assertEqual(result.shape[1], len(before))
            self.assertFalse(any("validationuniqueword" in term or "targetmetadata" in term for term in before))

    def test_optional_summary_and_empty_contrast_blocks(self):
        frame = pd.DataFrame({"Text": ["fresh tea", "great coffee", "sweet candy"]})
        estimator = self.estimator("separate_negation_contrast").fit(frame)
        self.assertEqual(estimator.block_sizes_["summary"], 0)
        self.assertEqual(estimator.block_sizes_["contrast_before"], 0)
        self.assertEqual(estimator.block_sizes_["contrast_after"], 0)
        result = estimator.transform(pd.DataFrame({"Text": ["great tea but costly"]}))
        self.assertEqual(result.shape[1], estimator.n_output_features_)

    def test_clone_and_serialization_preserve_features(self):
        frame = self.frame()
        for variant in VARIANTS:
            estimator = clone(self.estimator(variant))
            fitted = estimator.fit_transform(frame)
            stream = io.BytesIO()
            joblib.dump(estimator, stream)
            stream.seek(0)
            loaded = joblib.load(stream)
            np.testing.assert_allclose(fitted.toarray(), loaded.transform(frame).toarray())
            self.assertTrue(np.isfinite(fitted.data).all())
            self.assertTrue((fitted.data >= 0).all())

    def test_invalid_inputs_and_unfitted_transform(self):
        with self.assertRaises(ValueError):
            self.estimator("invalid").fit(self.frame())
        with self.assertRaises(ValueError):
            self.estimator("combined").fit(pd.DataFrame({"Score": [5]}))
        with self.assertRaises(NotFittedError):
            self.estimator("combined").transform(self.frame())


if __name__ == "__main__":
    unittest.main()
