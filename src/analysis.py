"""Create Milestone 2 figures and error statistics after running pipeline.py."""

import json
import re
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from pipeline import ARTIFACTS, LABELS, RES, ROOT, SEED

FIG = ROOT / "figures"
COL = {"negative": "#2a78d6", "neutral": "#eb6834", "positive": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LENGTH_BUCKETS = ["<30", "30-59", "60-119", "120+"]
STOP = set("""the a an and or but of to in on for with is it this that i my was
were be are as at have has had they them their you your we our so if from by
its it's i'm i've these those than there here what which who very just can
all one would will about me do did been get got when out up some more much
also only because than then any other into after over 's s t""".split())
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
}


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def top_from_coef(weights, names, k=10, eligible=None):
    """Rank eligible features without allowing excluded features into the list."""
    indices = np.arange(len(weights))
    if eligible is not None:
        indices = indices[np.asarray(eligible, dtype=bool)]
    order = np.argsort(-np.asarray(weights)[indices], kind="stable")
    return [(str(names[i]), float(weights[i])) for i in indices[order[:k]]]


def top_features(artifacts, min_count=100):
    svm = artifacts["models"]["Linear SVM"]
    tfidf_names = artifacts["tfidf"].get_feature_names_out()
    nb = artifacts["models"]["Multinomial NB"]
    bow_names = artifacts["bow"].get_feature_names_out()
    classes = list(nb.classes_)
    ratio = (nb.feature_log_prob_[classes.index("positive")]
             - nb.feature_log_prob_[classes.index("negative")])
    counts = np.asarray(artifacts["bow"].transform(artifacts["train_docs"]).sum(axis=0)).ravel()
    out = {"bow_min_train_occurrences": min_count}
    for label, suffix, direction in [("positive", "pos", 1), ("negative", "neg", -1)]:
        weights = svm.coef_[list(svm.classes_).index(label)]
        out[f"tfidf_svm_{suffix}"] = top_from_coef(weights, tfidf_names, eligible=weights > 0)
        weights = direction * ratio
        out[f"bow_nb_{suffix}"] = top_from_coef(
            weights, bow_names, eligible=(counts >= min_count) & (weights > 0))
    return out


def fig_top_features(features, figure_dir):
    for prefix, filename, title, axis_label in [
        ("tfidf_svm", "top_features.png", "TF-IDF + Linear SVM", "one-vs-rest coefficient"),
        ("bow_nb", "bow_top_features.png", "BoW + Multinomial NB", "log probability ratio"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        for ax, label, suffix in zip(axes, ["negative", "positive"], ["neg", "pos"]):
            selected = features[f"{prefix}_{suffix}"][::-1]
            ax.barh([word for word, _ in selected], [weight for _, weight in selected],
                    color=COL[label], height=0.6)
            ax.set_title(f"{label.capitalize()}: {title}", loc="left")
            ax.set_xlabel(axis_label)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0, labelcolor=INK)
        save_figure(fig, figure_dir / filename)


def fig_word_freq(artifacts, figure_dir):
    docs, labels = artifacts["train_docs"], artifacts["train_labels"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    tops = {}
    for ax, label in zip(axes, LABELS):
        selected = [doc for doc, lab in zip(docs, labels) if lab == label]
        counts = Counter()
        for doc in selected:
            counts.update(word for word in re.findall(r"[a-z']+", doc)
                          if word not in STOP and len(word) > 2)
        tops[label] = [(word, count / len(selected)) for word, count in counts.most_common(12)]
        top = tops[label][::-1]
        ax.barh([word for word, _ in top], [count for _, count in top], color=COL[label], height=0.6)
        ax.set_title(f"{label.capitalize()} (n={len(selected):,})", loc="left")
        ax.set_xlabel("occurrences per review")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    save_figure(fig, figure_dir / "word_freq.png")
    return tops


def sample_by_class(labels, per_class=700, seed=SEED):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in LABELS]
    return np.concatenate([rng.choice(group, min(per_class, len(group)), replace=False)
                           for group in groups])


def fig_svd(artifacts, figure_dir):
    X, labels = artifacts["Xtr_t"], np.asarray(artifacts["ytr"])
    n_components = min(10, *X.shape)
    if n_components < 2:
        raise ValueError("SVD analysis requires at least two training rows and features.")
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    projection = svd.fit_transform(X)
    indices = sample_by_class(labels)
    sample, sample_labels = projection[indices], labels[indices]
    names = artifacts["tfidf"].get_feature_names_out()
    terms = {str(k): [str(names[j]) for j in np.argsort(-np.abs(row))[:6]]
             for k, row in enumerate(svd.components_)}
    pairs = [(1, 2), (3, 4)] if n_components >= 5 else [(0, 1)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 3.0), squeeze=False)
    for ax, (i, j) in zip(axes.flat, pairs):
        for label in ["positive", "neutral", "negative"]:
            selected = sample_labels == label
            ax.scatter(sample[selected, i], sample[selected, j], s=8, alpha=0.55,
                       color=COL[label], label=label, edgecolors="white", linewidths=0.3)
        ax.set_xlabel(f"SVD component {i + 1}")
        ax.set_ylabel(f"SVD component {j + 1}")
        ax.set_title(f"TF-IDF components {i + 1} and {j + 1}", loc="left")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center", ncol=3,
               markerscale=2, bbox_to_anchor=(0.5, 1.06))
    save_figure(fig, figure_dir / "svd_scatter.png")
    means = {str(k): {label: float(projection[labels == label, k].mean())
                     if np.any(labels == label) else None for label in LABELS}
             for k in range(min(5, n_components))}
    return {"svd_var": svd.explained_variance_ratio_.tolist(),
            "svd_top_terms": terms, "svd_class_means": means}


def plot_confusion(ax, matrix, title):
    matrix = np.asarray(matrix)
    counts = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, counts, out=np.zeros_like(matrix, dtype=float), where=counts != 0)
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            percentage = f"{normalized[i, j]:.0%}" if counts[i, 0] else "n/a"
            ax.text(j, i, f"{matrix[i, j]:,}\n{percentage}", ha="center", va="center",
                    fontsize=8, color="white" if normalized[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def fig_confusion(summary, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, name in zip(axes, ["Multinomial NB", "Linear SVM"]):
        plot_confusion(ax, summary["confusion"][name], name)
    save_figure(fig, figure_dir / "confusion.png")
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    plot_confusion(ax, summary["confusion"]["Linear SVM"], "Linear SVM, test set")
    save_figure(fig, figure_dir / "confusion_svm.png")


def error_rate(values):
    return float(values.mean()) if len(values) else None


def length_statistics(reviews):
    buckets = pd.cut(reviews["n_words"], [0, 30, 60, 120, np.inf],
                     labels=LENGTH_BUCKETS, right=False)
    grouped = reviews.groupby(buckets, observed=True)["err"]
    counts = grouped.size().reindex(LENGTH_BUCKETS, fill_value=0)
    rates = grouped.mean().reindex(LENGTH_BUCKETS)
    return {bucket: [int(counts[bucket]), float(rates[bucket]) if counts[bucket] else None]
            for bucket in LENGTH_BUCKETS}


def error_analysis(predictions):
    reviews = predictions.copy()
    reviews["pred"] = reviews["pred_Linear_SVM"]
    reviews["err"] = reviews["pred"].ne(reviews["label"]).astype(int)
    text = reviews["Text"].fillna("").astype(str).str.lower().str.replace("’", "'", regex=False)
    reviews["contrast"] = text.str.contains(r"\b(?:but|however|although|though|unfortunately|except)\b")
    reviews["negation"] = text.str.contains(r"\b(?:not|never|no)\b|n't\b")
    wrong = reviews.loc[reviews["err"] == 1].copy()
    patterns = {}
    for marker in ["contrast", "negation"]:
        present = reviews.loc[reviews[marker], "err"]
        absent = reviews.loc[~reviews[marker], "err"]
        patterns[marker] = {"with": error_rate(present), "without": error_rate(absent),
                            "with_n": len(present), "without_n": len(absent),
                            "with_wrong": int(present.sum()), "without_wrong": int(absent.sum()),
                            "share": float(reviews[marker].mean()) if len(reviews) else None}
    by_length = length_statistics(reviews)
    nonpositive = length_statistics(reviews.loc[reviews["label"] != "positive"])
    class_errors = {label: reviews.loc[reviews["label"] == label, "err"] for label in LABELS}
    pairs = wrong.groupby(["label", "pred"]).size()
    statistics = {
        "n_wrong": len(wrong), "n_test": len(reviews), "by_len": by_length,
        "by_len_nonpos": {key: value[1] for key, value in nonpositive.items()},
        "by_len_nonpos_n": {key: value[0] for key, value in nonpositive.items()},
        "patterns": patterns,
        "err_by_true": {label: error_rate(values) for label, values in class_errors.items()},
        "n_by_true": {label: len(values) for label, values in class_errors.items()},
        "wrong_by_true": {label: int(values.sum()) for label, values in class_errors.items()},
        "pair_counts": {f"{true}->{pred}": int(count) for (true, pred), count in pairs.items()},
    }
    return statistics, wrong


def select_error_examples(wrong, count=8, seed=SEED):
    """Include confusion-pair variety, then fill the sample from remaining errors."""
    candidates = wrong.sort_values("Id", kind="stable").reset_index(drop=True)
    selected = [group.sample(n=1, random_state=seed).index[0]
                for _, group in candidates.groupby(["label", "pred"], sort=True)]
    selected = selected[:count]
    remaining = candidates.drop(index=selected)
    needed = min(count - len(selected), len(remaining))
    selected.extend(remaining.sample(n=needed, random_state=seed).index.tolist())
    return (candidates.loc[selected, ["Id", "Summary", "Text", "label", "pred"]]
            .rename(columns={"label": "true"}).reset_index(drop=True))


def fig_errors(statistics, figure_dir):
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    x = np.arange(len(LENGTH_BUCKETS))
    for offset, rates, color, label in [
        (-0.19, [statistics["by_len"][key][1] for key in LENGTH_BUCKETS], "#2a78d6", "all test reviews"),
        (0.19, list(statistics["by_len_nonpos"].values()), "#eb6834", "true negative/neutral only"),
    ]:
        values = [np.nan if value is None else value for value in rates]
        ax.bar(x + offset, values, 0.36, color=color, label=label)
        for position, value in zip(x + offset, rates):
            if value is not None:
                ax.text(position, value + 0.015, f"{value:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{key} words\n(n={statistics['by_len'][key][0]:,})" for key in LENGTH_BUCKETS])
    ax.set_ylabel("error rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.set_title("Linear SVM error rate by review length", loc="left")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, figure_dir / "error_by_length.png")


def fig_models(comparison, figure_dir):
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    y = np.arange(len(comparison))
    ax.barh(y, comparison["macro_f1"], color="#2a78d6", height=0.55)
    for position, (_, row) in zip(y, comparison.iterrows()):
        ax.text(row["macro_f1"] + 0.01, position,
                f"{row['macro_f1']:.3f}  (train {row['fit_seconds']:.1f}s)",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, comparison["model"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro F1 on held-out test (20%)")
    ax.set_title("Model comparison", loc="left")
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, figure_dir / "model_comparison.png")


def main():
    artifacts = joblib.load(ARTIFACTS)
    with (RES / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    predictions = pd.read_csv(RES / "test_predictions.csv")
    comparison = pd.read_csv(RES / "model_comparison.csv")
    FIG.mkdir(parents=True, exist_ok=True)
    out = top_features(artifacts)
    statistics, wrong = error_analysis(predictions)
    out["errors"] = statistics
    wrong.to_csv(RES / "misclassified_svm.csv", index=False)
    select_error_examples(wrong).to_csv(RES / "error_examples.csv", index=False)
    with plt.rc_context(STYLE):
        fig_top_features(out, FIG)
        out["word_freq"] = fig_word_freq(artifacts, FIG)
        out.update(fig_svd(artifacts, FIG))
        fig_confusion(summary, FIG)
        fig_errors(statistics, FIG)
        fig_models(comparison, FIG)
    with (RES / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, allow_nan=False)
    print(f"Wrote analysis for {statistics['n_test']:,} test reviews; "
          f"{statistics['n_wrong']:,} misclassified reviews.")


if __name__ == "__main__":
    main()
