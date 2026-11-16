"""CSE 3521 Milestone 2 - figures, top features, and error analysis.
Run after src/pipeline.py:   python src/analysis.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
LABELS = ["negative", "neutral", "positive"]

# Reference categorical palette, slots 1-3 (validated all-pairs, light mode)
COL = {"negative": "#2a78d6", "neutral": "#eb6834", "positive": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})

a = joblib.load(RES / "artifacts.joblib")
summary = json.load(open(RES / "summary.json"))
models, bow, tfidf = a["models"], a["bow"], a["tfidf"]
out = {}

# ------------------------------------------------ top-10 features
def top_from_coef(coef_row, names, k=10):
    i = np.argsort(coef_row)
    return [(names[j], float(coef_row[j])) for j in i[::-1][:k]]

# TF-IDF + Linear SVM: one-vs-rest weights for each class
svm = models["Linear SVM"]; names_t = tfidf.get_feature_names_out()
cls = list(svm.classes_)
out["tfidf_svm_pos"] = top_from_coef(svm.coef_[cls.index("positive")], names_t)
out["tfidf_svm_neg"] = top_from_coef(svm.coef_[cls.index("negative")], names_t)

# BoW + Multinomial NB: log P(w|pos) - log P(w|neg)
nb = models["Multinomial NB"]; names_b = bow.get_feature_names_out()
ncls = list(nb.classes_)
lr = nb.feature_log_prob_[ncls.index("positive")] - nb.feature_log_prob_[ncls.index("negative")]
# require some support so ultra-rare words do not dominate the ratio
counts = np.asarray(a["bow"].transform(a["train_docs"]).sum(axis=0)).ravel()
lr_masked = np.where(counts >= 100, lr, 0)
out["bow_nb_pos"] = top_from_coef(lr_masked, names_b)
out["bow_nb_neg"] = top_from_coef(-lr_masked, names_b)

def fig_top_features():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, lab, title in [(axes[0], "tfidf_svm_neg", "negative", "Top-10 negative (TF-IDF + SVM)"),
                                (axes[1], "tfidf_svm_pos", "positive", "Top-10 positive (TF-IDF + SVM)")]:
        feats = out[key][::-1]
        ax.barh([f for f, _ in feats], [w for _, w in feats], color=COL[lab], height=0.6)
        ax.set_title(title, loc="left"); ax.set_xlabel("SVM weight")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    fig.tight_layout(); fig.savefig(FIG / "top_features.png"); plt.close(fig)

# ------------------------------------------------ word frequency chart
STOP = set("""the a an and or but of to in on for with is it this that i my was
were be are as at have has had they them their you your we our so if from by
its it's i'm i've these those than there here what which who very just can
all one would will about me do did been get got when out up some more much
also only because than then any other into after over 's s t""".split())
def fig_word_freq():
    docs, labs = a["train_docs"], a["train_labels"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=False)
    tops = {}
    for ax, lab in zip(axes, LABELS):
        c = Counter()
        sel = [d for d, l in zip(docs, labs) if l == lab]
        for d in sel:
            c.update(w for w in re.findall(r"[a-z']+", d) if w not in STOP and len(w) > 2)
        n = len(sel)
        top = c.most_common(12)[::-1]
        tops[lab] = [(w, v / n) for w, v in top[::-1]]
        ax.barh([w for w, _ in top], [v / n for _, v in top], color=COL[lab], height=0.6)
        ax.set_title(f"{lab.capitalize()} (n={n:,})", loc="left")
        ax.set_xlabel("occurrences per review")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    fig.tight_layout(); fig.savefig(FIG / "word_freq.png"); plt.close(fig)
    out["word_freq"] = tops

# ------------------------------------------------ SVD (LSA) scatter
def fig_svd():
    X, y = a["Xtr_t"], a["ytr"]
    rng = np.random.default_rng(47)
    idx = np.concatenate([rng.choice(np.where(y == l)[0], 700, replace=False) for l in LABELS])
    svd = TruncatedSVD(n_components=10, random_state=47).fit(X)
    Z = svd.transform(X[idx]); yy = y[idx]
    names = tfidf.get_feature_names_out()
    tops = {k: [names[j] for j in np.argsort(-np.abs(svd.components_[k]))[:6]] for k in range(10)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, (i, j), ttl in [(axes[0], (1, 2), "Components 2 vs 3 (topic: coffee, dog food)"),
                            (axes[1], (3, 4), "Components 4 vs 5 (5 loads on 'great', 'love')")]:
        for lab in ["positive", "neutral", "negative"]:
            m = yy == lab
            ax.scatter(Z[m, i], Z[m, j], s=8, alpha=0.55, color=COL[lab], label=lab,
                       edgecolors=SURF, linewidths=0.3)
        ax.set_xlabel(f"SVD component {i+1}"); ax.set_ylabel(f"SVD component {j+1}")
        ax.set_title(ttl, loc="left"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, markerscale=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); fig.savefig(FIG / "svd_scatter.png"); plt.close(fig)
    out["svd_var"] = svd.explained_variance_ratio_.tolist()
    out["svd_top_terms"] = tops
    out["svd_class_means"] = {k: {l: float(svd.transform(X[y == l])[:, k].mean()) for l in LABELS} for k in range(5)}

# ------------------------------------------------ confusion matrices
def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, name in zip(axes, ["Multinomial NB", "Linear SVM"]):
        cm = np.array(summary["confusion"][name])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(name, loc="left")
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(FIG / "confusion.png"); plt.close(fig)

# ------------------------------------------------ error analysis
def error_analysis():
    te = pd.read_csv(RES / "test_predictions.csv")
    pred = te["pred_Linear_SVM"]; wrong = te[pred != te["label"]].copy()
    wrong["pred"] = pred[pred != te["label"]]
    te["err"] = (pred != te["label"]).astype(int)

    # Pattern 1: length buckets
    bins = [0, 30, 60, 120, 10000]; lab = ["<30", "30-59", "60-119", "120+"]
    te["len_bucket"] = pd.cut(te["n_words"], bins=bins, labels=lab, right=False)
    by_len = te.groupby("len_bucket", observed=True).agg(n=("err", "size"), err=("err", "mean"))
    # error rate restricted to non-positive true labels (where it matters)
    np_ = te[te.label != "positive"]
    by_len_np = np_.groupby("len_bucket", observed=True)["err"].mean()

    # Pattern 2: contrast / negation markers
    txt = te["Text"].str.lower()
    te["contrast"] = txt.str.contains(r"\b(?:but|however|although|though|unfortunately|except)\b")
    te["negation"] = txt.str.contains(r"\b(?:not|n't|never|no)\b|n't")
    pat = {}
    for col in ["contrast", "negation"]:
        pat[col] = {"with": float(te[te[col]].err.mean()), "without": float(te[~te[col]].err.mean()),
                    "share": float(te[col].mean())}
    # Pattern 3: neutral is the hardest class
    err_by_true = te.groupby("label")["err"].mean().to_dict()
    # Pattern 4: positive-text but low star / negative text high star (label noise)
    out["errors"] = dict(n_wrong=int(len(wrong)), n_test=int(len(te)),
                         by_len={k: [int(v.n), float(v.err)] for k, v in by_len.iterrows()},
                         by_len_nonpos={str(k): float(v) for k, v in by_len_np.items()},
                         patterns=pat, err_by_true=err_by_true,
                         pair_counts=wrong.groupby(["label", "pred"]).size().astype(int)
                         .rename(lambda x: x).to_dict())
    out["errors"]["pair_counts"] = {f"{k[0]}->{k[1]}": int(v)
                                    for k, v in wrong.groupby(["label", "pred"]).size().items()}

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(lab)); w = 0.36
    ax.bar(x - w/2 - 0.01, by_len["err"].values, w, color="#2a78d6", label="all test reviews")
    ax.bar(x + w/2 + 0.01, by_len_np.values, w, color="#eb6834", label="true negative/neutral only")
    for xi, v in zip(x - w/2, by_len["err"].values): ax.text(xi, v + 0.005, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    for xi, v in zip(x + w/2, by_len_np.values): ax.text(xi, v + 0.005, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{l} words\n(n={n:,})" for l, n in zip(lab, by_len["n"])])
    ax.set_ylabel("error rate"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.set_title("Linear SVM error rate by review length", loc="left")
    ax.set_ylim(0, 0.5); ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "error_by_length.png"); plt.close(fig)

    wrong["contrast"] = te.loc[wrong.index, "contrast"]
    wrong.to_csv(RES / "misclassified_svm.csv", index=False)

def fig_models():
    d = pd.read_csv(RES / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    y = np.arange(len(d))
    ax.barh(y, d["macro_f1"], color="#2a78d6", height=0.55)
    for yi, (f, t) in enumerate(zip(d["macro_f1"], d["fit_seconds"])):
        ax.text(f + 0.01, yi, f"{f:.3f}  (train {t:.1f}s)", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, d["model"]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("macro F1 on held-out test (20%)")
    ax.set_title("Model comparison", loc="left"); ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png"); plt.close(fig)

fig_top_features(); fig_word_freq(); fig_svd(); fig_confusion(); error_analysis(); fig_models()
json.dump(out, open(RES / "analysis.json", "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "word_freq"}, indent=1, default=str))
