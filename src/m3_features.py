"""Data-engineering feature candidates. No classifier training or predictions.

Keep the existing M2 cleaner and negation experiment as the compatibility base.
Fit this transformer INSIDE each training/CV fold, not before cross-validation.
Input is a DataFrame with raw Text and an optional raw Summary column.
"""

import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from pipeline import clean
from experiments import neg_scope


VARIANTS = ("combined", "separate", "separate_negation", "separate_negation_contrast")
# Intentionally bounded rule, not a parser or a claim about the overall verdict.
CONTRAST = re.compile(r"\b(?:but|however|although|though|yet|nevertheless)\b", re.I)
WORD = re.compile(r"\w", re.UNICODE)


def contrast_parts(text):
    """Return spans around the last eligible contrast marker, or two blanks.

    A leading 'although/though' uses its first comma as the boundary. Else use
    the last marker having text on both sides. Ignore additive 'not only ... but
    also' and idioms 'nothing/anything/everything/all but'. Labels say before and
    after, never positive or negative: either side may contain the main opinion.
    """
    text = clean(text)
    matches = list(CONTRAST.finditer(text))
    for match in reversed(matches):
        before, after = text[:match.start()].strip(), text[match.end():].strip()
        marker = match.group().lower()
        if marker == "but":
            if re.search(r"\b(?:nothing|anything|everything|all)\s*$", before):
                continue
            if re.search(r"\bnot\s+only\b", before) and re.match(r"also\b", after):
                continue
        if marker in {"although", "though"} and not before:
            if "," not in after:
                continue
            before, after = (part.strip() for part in after.split(",", 1))
        if WORD.search(before) and WORD.search(after):
            return before, after
    return "", ""


def raw_fields(frame):
    """Select text fields only. Never infer features from Score, label or IDs."""
    if not isinstance(frame, pd.DataFrame) or "Text" not in frame:
        raise ValueError("Expected a pandas DataFrame with a Text column.")
    def string_field(name):
        if name not in frame:
            return pd.Series("", index=frame.index, dtype="object")
        return frame[name].fillna("").astype(str)
    return string_field("Summary"), string_field("Text")


def feature_blocks(frame, variant):
    """Create named text channels with deterministic, non-learned rules."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose one of {VARIANTS}")
    summary, body = raw_fields(frame)
    if variant == "combined":
        return {"combined": (summary + " . " + body).map(clean).tolist()}
    if variant == "separate":
        return {"summary": summary.map(clean).tolist(), "body": body.map(clean).tolist()}
    blocks = {
        "summary": summary.map(lambda value: neg_scope(clean(value))).tolist(),
        "body": body.map(lambda value: neg_scope(clean(value))).tolist(),
    }
    if variant == "separate_negation_contrast":
        spans = body.map(contrast_parts)
        # Independent scopes in the two new channels prevent carry-over across
        # their boundary. The original full-body negation channel is unchanged.
        blocks["contrast_before"] = [neg_scope(pair[0]) for pair in spans]
        blocks["contrast_after"] = [neg_scope(pair[1]) for pair in spans]
    return blocks


class ReviewFeatureTransformer(TransformerMixin, BaseEstimator):
    """Cloneable TF-IDF transformer for all four data-engineering candidates.

    Parameters match M2's TF-IDF settings by default. Separate channels keep
    their own training-only vocabularies/IDFs. Channel vectors are concatenated
    without another normalization, matching M2's separate-feature experiments.
    Empty/pruned channels have zero columns; transform never grows vocabulary.
    """

    def __init__(self, variant="combined", min_df=2, max_df=0.9, sublinear_tf=True):
        self.variant = variant
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        self._fit_blocks(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self._fit_blocks(X)

    def _fit_blocks(self, X):
        blocks = feature_blocks(X, self.variant)
        self.vectorizers_ = {}
        self.block_sizes_ = {}
        matrices = []
        for name, documents in blocks.items():
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=self.min_df, max_df=self.max_df,
                sublinear_tf=self.sublinear_tf,
            )
            # Empty contrast channels are legitimate on a training fold with no
            # matching clauses. Other configuration/input errors still propagate.
            try:
                matrix = vectorizer.fit_transform(documents).tocsr()
            except ValueError as error:
                if not any(message in str(error) for message in (
                    "empty vocabulary", "After pruning, no terms remain"
                )):
                    raise
                vectorizer = None
                matrix = sparse.csr_matrix((len(X), 0), dtype=np.float64)
            self.vectorizers_[name] = vectorizer
            self.block_sizes_[name] = matrix.shape[1]
            matrices.append(matrix)
        self.n_output_features_ = sum(self.block_sizes_.values())
        if not self.n_output_features_:
            raise ValueError("No usable training vocabulary in any channel.")
        self.n_fit_rows_ = len(X)
        return sparse.hstack(matrices, format="csr")

    def transform(self, X):
        check_is_fitted(self, "vectorizers_")
        blocks = feature_blocks(X, self.variant)
        matrices = [
            vectorizer.transform(blocks[name]) if vectorizer is not None
            else sparse.csr_matrix((len(X), 0), dtype=np.float64)
            for name, vectorizer in self.vectorizers_.items()
        ]
        return sparse.hstack(matrices, format="csr")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "vectorizers_")
        return np.array([
            f"{name}__{term}"
            for name, vectorizer in self.vectorizers_.items()
            if vectorizer is not None
            for term in vectorizer.get_feature_names_out()
        ], dtype=object)
