"""Build Milestone2_Report.pdf from the current M2 results and figures."""
import html
import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"


def read_json(name):
    with (RES / name).open(encoding="utf-8") as stream:
        return json.load(stream)


S = read_json("summary.json")
A = read_json("analysis.json")
M = pd.read_csv(RES / "model_comparison.csv").set_index("model")
X = pd.read_csv(RES / "experiments.csv").set_index("variant")
EXAMPLES = pd.read_csv(RES / "error_examples.csv").fillna("")
E = A["errors"]
svm, nb, maj, lr = M.loc["Linear SVM"], M.loc["Multinomial NB"], M.loc["Majority class"], M.loc["Logistic Regression"]
Q = S["data_quality"]
BOW = S["feature_settings"]["BoW"]
TFIDF = S["feature_settings"]["TF-IDF"]
SVM_SETTINGS = S["model_settings"]["Linear SVM"]

# Prevent a stale selection from being presented as errors from the current run.
predictions = pd.read_csv(RES / "test_predictions.csv").set_index("Id")
if len(predictions) != S["n_test"] or E["n_test"] != S["n_test"]:
    raise ValueError("Results describe different test splits; rerun pipeline.py and analysis.py.")
for example in EXAMPLES.itertuples(index=False):
    if example.Id not in predictions.index:
        raise ValueError(f"Error example {example.Id} is missing from the current test split.")
    current = predictions.loc[example.Id]
    if (example.true != current["label"] or example.pred != current["pred_Linear_SVM"]
            or example.true == example.pred):
        raise ValueError(f"Error example {example.Id} does not match the current predictions.")

ss = getSampleStyleSheet()
INK, MUTED, RULE, HEAD = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d3"), colors.HexColor("#f1f0ec")
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.4, leading=12.4, textColor=INK, spaceAfter=4)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)
cap = ParagraphStyle("c", parent=body, fontSize=8, leading=10, textColor=MUTED, spaceAfter=8)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=9.6, spaceBefore=4, spaceAfter=2)
title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=2)
todo = ParagraphStyle("todo", parent=body, backColor=colors.HexColor("#fff4d6"), borderPadding=3)

P = lambda t, s=body: Paragraph(t, s)
pct = lambda v: "n/a" if pd.isna(v) else f"{100*v:.2f}%"
f3 = lambda v: f"{v:.3f}"


def escaped(text):
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text)))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return html.escape(text.encode("cp1252", "replace").decode("cp1252"))


def excerpt(text, limit=220):
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text)))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        half = (limit - 5) // 2
        text = text[:half] + " ... " + text[-half:]
    return escaped(text)


def feature_names(key, count=5):
    return ", ".join(escaped(name) for name, _ in A[key][:count])


def table(rows, widths, header=True, size=8, align_right_from=1):
    data = [[Paragraph(str(c), ParagraphStyle("tc", parent=small, fontSize=size, leading=size + 2,
             fontName="Helvetica-Bold" if (header and i == 0) else "Helvetica")) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.6, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.6, INK),
          ("LINEABOVE", (0, 0), (-1, 0), 0.6, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st)); return t


def img(name, w):
    from PIL import Image as PI
    with PI.open(FIG / name) as image:
        iw, ih = image.size
    return Image(str(FIG / name), width=w, height=w * ih / iw)


story = []
story += [P("AI-Powered Text Intelligence Platform: Milestone 2", title),
          P("CSE 3521 Survey of AI I, Autumn 2026 &nbsp;|&nbsp; Group Phi &nbsp;|&nbsp; Sean Kong, Nicholas Randolph, Kyle Baumgartner, Mike Ji, Ian Roy "
            "&nbsp;|&nbsp; Repo: <link href='https://github.com/SeanK000/CSE3521-AI-Project' color='#2a78d6'>"
            "github.com/SeanK000/CSE3521-AI-Project</link>", cap)]

story.append(P(
    f"<b>Summary.</b> This milestone builds two text representations and initial classical classifiers. The app currently uses a "
    f"<b>linear Support Vector Machine (SVM) with TF-IDF features</b>. The pipeline (<font face='Courier'>src/pipeline.py</font>) "
    f"cleans and de-duplicates the {S['n_raw']:,}-review sample, then uses a stratified "
    f"{1-S['test_fraction']:.0%}/{S['test_fraction']:.0%} train/test split (seed {S['seed']}). "
    f"The SVM reaches {pct(svm.accuracy)} accuracy and "
    f"{f3(svm.macro_f1)} macro-F1, against {pct(maj.accuracy)} / {f3(maj.macro_f1)} for a majority-class baseline that always predicts "
    f"\"positive\" in this run. Classifier fitting takes {svm.fit_seconds:.2f} seconds. "
    f"Neutral reviews remain its main weakness (F1 {f3(svm.neutral_f1)})."))

# ------------------------------------------------------------------ 1
story.append(P("1. Feature engineering", h1))
story.append(P(
    f"<b>Cleaning.</b> The pipeline validates ratings (1-5) and removes empty cleaned bodies. This sample has "
    f"{Q['invalid_scores_removed']:,} invalid ratings and {Q['empty_text_removed']:,} empty bodies removed. "
    f"De-duplication removes {Q['exact_duplicates_removed']:,} exact repeats and "
    f"{Q['cleaning_equivalent_duplicates_removed']:,} additional cleaning-equivalent bodies "
    f"({Q['duplicates_removed']:,} total), before splitting. It keeps the first occurrence; "
    f"conflicting-label duplicate-body groups: {Q['conflicting_duplicate_groups']:,}. "
    f"This leaves {S['n_dedup']:,} reviews ({S['n_train']:,} train / {S['n_test']:,} test). "
    f"Cleaning decodes HTML entities, strips tags, replaces URLs, lower-cases, and normalizes whitespace. "
    f"The {Q['missing_summary']:,} missing summaries become empty strings, and summary and body are joined. "
    f"Cleaned bodies average {S['mean_words']:.0f} words (median {S['median_words']:.0f}). "
    f"No stop-word list is used for modeling, preserving negation words such as <i>not</i> and <i>never</i>."))

story.append(table([
    ["Representation", "Method", "Size"],
    ["Bag of Words", f"Counts of word n-grams ({BOW['ngram_range'][0]}-{BOW['ngram_range'][1]}); "
     f"min_df={BOW['min_df']}. A frequency baseline used by Multinomial NB.", f"{S['bow_vocab']:,} features"],
    ["TF-IDF", f"Word n-grams ({TFIDF['ngram_range'][0]}-{TFIDF['ngram_range'][1]}); "
     f"min_df={TFIDF['min_df']}, max_df={TFIDF['max_df']}. Sublinear term frequency is enabled "
     f"({TFIDF['sublinear_tf']}); inverse document frequency downweights widespread terms. "
     f"Vectors use {escaped(TFIDF['norm'])} normalization. Bigrams retain phrases such as 'not good'.", f"{S['tfidf_vocab']:,} features"],
    ["LSA (analysis only)", "Truncated SVD of training TF-IDF, used only to visualize the feature space (Figure 4).",
     f"{len(A['svd_var'])} dims"],
], [1.25 * inch, 4.75 * inch, 1.0 * inch]))
story.append(P("Table 1. Text representations, fitted on the training split only.", cap))

story.append(KeepTogether([P("<b>Top-10 most informative features.</b> For a linear SVM, a feature's weight in the one-vs-rest "
    "classifier for a class measures how strongly it pushes a review toward that class.", body),
    img("top_features.png", 5.8 * inch),
    P("Figure 1. Top-10 TF-IDF features with the largest SVM weights for the positive and negative classes.", cap)]))
story.append(P(
    f"The leading positive SVM terms are <i>{feature_names('tfidf_svm_pos')}</i>; the leading negative terms are "
    f"<i>{feature_names('tfidf_svm_neg')}</i>. These coefficients describe training associations, not probabilities or "
    f"causal explanations of individual predictions."))
story.append(KeepTogether([img("bow_top_features.png", 5.8 * inch),
    P(f"Figure 2. Top-10 BoW features ranked by Multinomial NB log P(term|positive) - log P(term|negative), "
      f"and the reverse ratio for negative features. The display requires at least "
      f"{A['bow_min_train_occurrences']} total training token occurrences per term; it does not alter training.", cap)]))
story.append(img("word_freq.png", 5.6 * inch))
story.append(P("Figure 3. Token occurrences per training review, using combined summary/body text by class. "
               "Common function words are removed only for display; repeated tokens can make a value exceed one.", cap))
story.append(img("svd_scatter.png", 5.4 * inch))
v = A["svd_var"]
story.append(P(
    f"Figure 4. Truncated SVD fitted to training TF-IDF, displaying a sample of training reviews (seed {S['seed']}). "
    f"The first {min(5, len(v))} components explain {pct(sum(v[:5]))} of variance. The classifier uses all "
    f"{S['tfidf_vocab']:,} sparse features; this low-dimensional projection is for visualization only.", cap))

# ------------------------------------------------------------------ 2
story.append(P("2. Baseline results", h1))
story.append(P(
    f"The initial SVM uses {S['n_train']:,} training reviews and the held-out {S['test_fraction']:.0%} "
    f"({S['n_test']:,} reviews: {S['test_label_counts']['positive']:,} positive, "
    f"{S['test_label_counts']['negative']:,} negative, {S['test_label_counts']['neutral']:,} neutral). "
    f"The model is <font face='Courier'>LinearSVC</font>, with C={SVM_SETTINGS['C']} and "
    f"class_weight={escaped(SVM_SETTINGS['class_weight'])}. These are fixed initial settings. "
    f"The optional <font face='Courier'>src/tune.py</font> compares settings using training-only three-fold CV, "
    f"refitting vocabulary and IDF in each fold; it does not automatically change the initial model. "
    f"Positive reviews are {pct(S['label_counts']['positive']/S['n_dedup'])} of retained data, so "
    f"<b>macro-averaged</b> scores give each class equal weight."))

tc = S["test_label_counts"]
rows = [["Class", "Precision", "Recall", "F1", "Support"]]
for lab in S["labels"]:
    rows.append([lab, f3(svm[f"{lab}_precision"]), f3(svm[f"{lab}_recall"]),
                 f3(svm[f"{lab}_f1"]), f"{tc[lab]:,}"])
rows.append(["<b>macro average</b>", f"<b>{f3(svm.macro_precision)}</b>",
             f"<b>{f3(svm.macro_recall)}</b>", f"<b>{f3(svm.macro_f1)}</b>", f"{S['n_test']:,}"])
rows.append(["<b>accuracy</b>", "", "", f"<b>{pct(svm.accuracy)}</b>", f"{S['n_test']:,}"])
cls_tab = table(rows, [1.05 * inch, 0.65 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch])
cmp_tab = table([
    ["", "Majority class", "Linear SVM"],
    ["Accuracy", pct(maj.accuracy), pct(svm.accuracy)],
    ["Macro precision", f3(maj.macro_precision), f3(svm.macro_precision)],
    ["Macro recall", f3(maj.macro_recall), f3(svm.macro_recall)],
    ["Macro F1", f3(maj.macro_f1), f3(svm.macro_f1)],
    ["Negative recall", pct(maj.negative_recall), pct(svm.negative_recall)],
], [1.05 * inch, 0.85 * inch, 0.85 * inch])
left = [cls_tab, P("Table 2. SVM per-class test results.", cap), cmp_tab,
        P("Table 3. SVM versus the majority-class baseline.", cap)]
grid = Table([[left, [img("confusion_svm.png", 2.8 * inch),
                     P("Figure 5. SVM confusion matrix: counts and percentages within each true class.", cap)]]],
             colWidths=[3.75 * inch, 3.25 * inch], hAlign="LEFT")
grid.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
]))
story.append(grid)
story.append(P(
    f"<b>Is the model learning?</b> The majority baseline scores {pct(maj.accuracy)} accuracy but only "
    f"{f3(maj.macro_f1)} macro-F1, with zero negative and neutral recall. The SVM raises accuracy by "
    f"{100*(svm.accuracy-maj.accuracy):.2f} points and macro-F1 by {svm.macro_f1-maj.macro_f1:.3f}. "
    f"It recovers {pct(svm.negative_recall)} of negative reviews. Neutral recall remains "
    f"{pct(svm.neutral_recall)}, versus {pct(svm.positive_recall)} for positive reviews."))

rows = [["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Fit (s)"]]
for name, model in M.iterrows():
    rows.append([escaped(name), pct(model.accuracy), f3(model.macro_precision),
                 f3(model.macro_recall), f3(model.macro_f1), f"{model.fit_seconds:.2f}"])
story.append(table(rows, [1.8 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch, 1.1 * inch, 1.0 * inch]))
story.append(P("Table 4. Initial models on the same test split. P and R denote precision and recall.", cap))
story.append(P(
    f"<b>Current app model.</b> {escaped(M['accuracy'].idxmax())} has the highest accuracy in this run. "
    f"Logistic Regression has macro-F1 "
    f"{f3(lr.macro_f1)}, versus {f3(svm.macro_f1)} for SVM. SVM fitting takes {svm.fit_seconds:.2f}s, versus "
    f"{lr.fit_seconds:.2f}s for LR; SVM prediction takes {svm.predict_ms_per_1k:.3f}ms per 1,000 reviews. "
    f"These timings exclude feature extraction and vary by machine. The Streamlit app loads saved TF-IDF "
    f"and SVM from <font face='Courier'>models/artifacts.joblib</font>; it does not retrain on each submission."))

if (RES / "tuning.csv").is_file():
    tuning = pd.read_csv(RES / "tuning.csv")
    if not tuning["n_training_reviews"].eq(S["n_train"]).all():
        raise ValueError("Tuning results use a different training size; rerun tune.py.")
    best = tuning.loc[tuning.groupby("model")["mean_macro_f1"].idxmax()]
    comparisons = "; ".join(
        f"{escaped(row.model)}: C={row.C:g}, mean macro-F1={f3(row.mean_macro_f1)}"
        for row in best.itertuples(index=False)
    )
    story.append(P(f"<b>Training-only tuning.</b> Best tested settings in results/tuning.csv: {comparisons}. "
                   "These are CV validation averages, separate from the test scores above.", small))

# ------------------------------------------------------------------ 3
story.append(P("3. Error analysis", h1))
story.append(P(
    f"The SVM misclassifies {E['n_wrong']:,} of {E['n_test']:,} test reviews. Error rates by true class are "
    f"{pct(E['err_by_true']['positive'])} for positive, {pct(E['err_by_true']['negative'])} for negative, and "
    f"{pct(E['err_by_true']['neutral'])} for neutral. The following {len(EXAMPLES)} examples come from "
    f"<font face='Courier'>results/error_examples.csv</font>. They are selected deterministically across "
    f"confusion pairs, then filled from remaining errors (seed {S['seed']}); the full reviews retain their "
    f"original IDs."))

rows = [["ID / Summary", "Review excerpt", "True", "Predicted"]]
for example in EXAMPLES.itertuples(index=False):
    rows.append([f"{escaped(example.Id)}<br/>{excerpt(example.Summary, 70)}",
                 excerpt(example.Text), escaped(example.true), escaped(example.pred)])
story.append(table(rows, [1.5 * inch, 4.1 * inch, 0.7 * inch, 0.7 * inch], size=7.6))
story.append(P(f"Table 5. {len(EXAMPLES)} actual SVM errors. Long excerpts retain their beginning and ending; "
               "full text is in the CSV.", cap))

pc, pn = E["patterns"]["contrast"], E["patterns"]["negation"]
story.append(P(
    f"<b>Measured pattern: contrast-marked bodies.</b> Reviews containing <i>but / however / although / "
    f"though / unfortunately / except</i> have error rate {pct(pc['with'])} "
    f"({pc['with_wrong']:,}/{pc['with_n']:,}), versus {pct(pc['without'])} "
    f"({pc['without_wrong']:,}/{pc['without_n']:,}) without those markers. These are descriptive associations; "
    f"the groups also differ in length and class mix. A candidate explanation to check in the full reviews "
    f"is that short word n-grams miss which clause carries the overall verdict."))
length_rates = "; ".join(
    f"{escaped(bucket)} words: {pct(values[1])} (n={values[0]:,})"
    for bucket, values in E["by_len"].items()
)
story.append(P(
    f"<b>Negation and length.</b> Bodies with negation markers have error rate {pct(pn['with'])} "
    f"({pn['with_wrong']:,}/{pn['with_n']:,}), versus {pct(pn['without'])} "
    f"({pn['without_wrong']:,}/{pn['without_n']:,}) without them. These groups overlap the contrast groups. "
    f"Body-length rates are {length_rates}. Figure 6 also separates true negative/neutral reviews; "
    f"the 120+ bucket has no upper length limit."))
story.append(KeepTogether([
    img("error_by_length.png", 3.6 * inch),
    P("Figure 6. SVM error by cleaned body word count, for all reviews and true negative/neutral reviews.", cap),
]))

if not X["evaluation_split"].eq("training-only validation").all():
    raise ValueError("Experiment scores must use training-only validation; rerun experiments.py.")
if not (X["n_train"] + X["n_validation"]).eq(S["n_train"]).all():
    raise ValueError("Experiment results use a different training pool; rerun experiments.py.")
xb = X.loc["baseline: summary+text in one TF-IDF"]
xs = X.loc["separate summary & body TF-IDF"]
xn = X.loc["separate + negation scoping"]
story.append(P(
    f"<b>Feature experiments.</b> <font face='Courier'>src/experiments.py</font> uses {int(xb.n_train):,} inner "
    f"training reviews and {int(xb.n_validation):,} validation reviews, all from the outer training pool. "
    f"It excludes the held-out test set. Combined features score {f3(xb.macro_f1)} macro-F1 / "
    f"{f3(xb.neutral_f1)} neutral F1; separate summary/body features score {f3(xs.macro_f1)} / "
    f"{f3(xs.neutral_f1)}; separate features plus punctuation-bounded negation scoping score "
    f"{f3(xn.macro_f1)} / {f3(xn.neutral_f1)}. "
    f"These validation scores are separate from the test results above; the app still uses the initial "
    f"combined-feature SVM. A next check is training-only CV of these variants and a contrast-clause "
    f"feature, before adopting a change."))
best_macro = escaped(X["macro_f1"].idxmax())
best_neutral = escaped(X["neutral_f1"].idxmax())
story.append(P(
    f"The highest validation macro-F1 in this comparison is from <i>{best_macro}</i>; "
    f"the highest neutral F1 is from <i>{best_neutral}</i>. The feature choice therefore depends on "
    f"which class tradeoff the team prioritizes.", small))

# ------------------------------------------------------------------ 4
story.append(P("4. Updated timeline", h1))
story.append(table([
    ["Milestone", "Due", "Plan (revised)", "Update"],
    ["M2", "Nov 15", "BoW and TF-IDF, initial classical models, majority comparison, error analysis.",
     "Cleaned-body deduplication and training-only validation; saved-model SVM app available locally."],
    ["M3", "Nov 22", "Validate feature variants; add a deep/modern model with loss curves; compare models; extend the local app.",
     "Evaluate contrast/negation features on training data; keep minority-class performance visible."],
    ["M4 report", "Nov 29", "Finalize evaluation, report and slides; complete deployment with predictions from at least two models.",
     "Final report and slides use the November 29 assignment deadline."],
    ["M4 demo", "See Carmen", "Live demonstration and presentation; each member speaks.",
     "Confirm the demo schedule on Carmen."],
], [0.75 * inch, 0.65 * inch, 3.1 * inch, 2.5 * inch]))
story.append(Spacer(1, 4))
story.append(P(
    f"<b>Scope and rationale.</b> The M2 feature pipeline retains three-class sentiment and both text "
    f"representations. SVM is used in the current app, while NB and LR remain comparison models. "
    f"The SVM minus Multinomial NB macro-F1 difference is {svm.macro_f1-nb.macro_f1:.3f}; "
    f"{escaped(M['macro_f1'].idxmax())} has the highest macro-F1 in this run. Follow-up feature comparisons use validation "
    f"data rather than the outer test set. The deep/modern model and two-model deployment remain "
    f"later-milestone work."))


# ------------------------------------------------------------------ 5
story.append(P("5. Individual progress log", h1))
# remember to add your progress here :)
log = [
    ("Sean Kong (Project Lead)", "[Completed since M1: e.g., coordinated M2, set up repo structure and evaluation protocol, wrote report sections.] [Next: e.g., M3 planning, cross-validation of fixes.]"),
    ("Nicholas Randolph (App Engineer)", "[Completed since M1: ...] [Next: e.g., web-app skeleton that loads saved models and shows side-by-side predictions.]"),
    ("Kyle Baumgartner (Data Engineer)", "[Completed since M1: e.g., cleaning, de-duplication, and the train/test split.] [Next: e.g., negation scoping and contrast-clause features.]"),
    ("Mike Ji (Analyst / Model Trainer)", "[Completed since M1: split the data, train the SVM, tuning, and analyze them. Completes what is left in the streamlitApp] [Next: Tuning for neutral.]"),
    ("Ian Roy (ML Engineer)", "[Completed since M1: e.g., TF-IDF + SVM model and hyperparameter search.] [Next: e.g., embedding / DistilBERT model for M3.]"),
]
for who, txt in log:
    story.append(P(f"<b>{who}.</b> {txt}", todo))


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    c.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"CSE 3521 Milestone 2  |  page {d.page}")
    c.restoreState()

doc = SimpleDocTemplate(str(ROOT / "Milestone2_Report.pdf"), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.65 * inch, title="CSE 3521 Milestone 2", author="Team (Sean Kong et al.)")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("pages:", doc.page)
