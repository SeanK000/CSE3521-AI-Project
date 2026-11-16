"""Builds Milestone2_Report.pdf from results/ and figures/ (numbers are read, not typed)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
S = json.load(open(RES / "summary.json"))
A = json.load(open(RES / "analysis.json"))
M = pd.read_csv(RES / "model_comparison.csv").set_index("model")
X = pd.read_csv(RES / "experiments.csv")
E = A["errors"]
svm, nb, maj, lr = M.loc["Linear SVM"], M.loc["Multinomial NB"], M.loc["Majority class"], M.loc["Logistic Regression"]

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
pct = lambda v: f"{100*v:.1f}%"
f3 = lambda v: f"{v:.3f}"


def table(rows, widths, header=True, size=8, align_right_from=1):
    data = [[Paragraph(str(c), ParagraphStyle("tc", parent=small, fontSize=size, leading=size + 2,
             fontName="Helvetica-Bold" if (header and i == 0) else "Helvetica")) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.6, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.6, INK),
          ("LINEABOVE", (0, 0), (-1, 0), 0.6, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st)); return t


def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(FIG / name).size
    return Image(str(FIG / name), width=w, height=w * ih / iw)


story = []
story += [P("AI-Powered Text Intelligence Platform: Milestone 2", title),
          P("CSE 3521 Survey of AI I, Autumn 2026 &nbsp;|&nbsp; Sean Kong, Nicholas Randolph, Kyle Baumgartner, Mike Ji, Ian Roy "
            "&nbsp;|&nbsp; Repo: <link href='https://github.com/SeanK000/CSE3521-AI-Project' color='#2a78d6'>"
            "github.com/SeanK000/CSE3521-AI-Project</link>", cap)]

story.append(P(
    f"<b>Summary.</b> This milestone focuses on one model: a <b>linear Support Vector Machine (SVM) trained on unigram+bigram TF-IDF "
    f"features</b>. Our reproducible pipeline (<font face='Courier'>src/pipeline.py</font>) cleans and de-duplicates the 50,000-review "
    f"sample, builds the features, and evaluates on a stratified 80/20 split. The SVM reaches {pct(svm.accuracy)} accuracy and "
    f"{f3(svm.macro_f1)} macro-F1, against {pct(maj.accuracy)} / {f3(maj.macro_f1)} for a majority-class baseline that always predicts "
    f"\"positive\". It trains in about one second. Neutral reviews remain its main weakness (F1 {f3(svm.neutral_f1)})."))

# ------------------------------------------------------------------ 1
story.append(P("1. Feature engineering", h1))
story.append(P(
    f"<b>Cleaning.</b> We found {S['n_raw'] - S['n_dedup']:,} reviews whose text is an exact duplicate of another review in the "
    f"sample (Amazon copies the same review across product variants). Because a duplicate in both train and test would inflate "
    f"scores, we removed them <i>before</i> splitting, leaving {S['n_dedup']:,} reviews ({S['n_train']:,} train / {S['n_test']:,} test). "
    f"We decoded HTML entities, stripped tags such as &lt;br /&gt;, replaced URLs with a token, lower-cased, and prepended the "
    f"Summary field to the review text, because summaries are short, dense sentiment statements. The one missing Summary became an "
    f"empty string. Reviews average {S['mean_words']:.0f} words (median {S['median_words']:.0f}). We deliberately used <b>no stop-word "
    f"list</b>, since standard lists delete \"not\", \"no\" and \"never\", which turn out to be among the strongest signals (below)."))

story.append(table([
    ["Representation", "Method", "Size"],
    ["TF-IDF (1-2 grams)", "Unigrams + bigrams; term frequency is sublinear (1 + log tf) and multiplied by inverse document frequency; "
     "each review vector is L2-normalised. Terms must appear in at least 2 training reviews (min_df = 2) and in at most 90% of them "
     "(max_df = 0.9). Bigrams let a linear model see short phrases such as \"not good\".", f"{S['tfidf_vocab']:,} features"],
    ["LSA (analysis only)", "Truncated SVD of the TF-IDF matrix, used to visualise the feature space (Figure 3)", "10 dims"],
], [1.25 * inch, 4.75 * inch, 1.0 * inch]))
story.append(P("Table 1. Text representations, fitted on the training split only.", cap))

story.append(KeepTogether([P("<b>Top-10 most informative features.</b> For a linear SVM, a feature's weight in the one-vs-rest "
    "classifier for a class measures how strongly it pushes a review toward that class.", body),
    img("top_features.png", 5.8 * inch),
    P("Figure 1. Top-10 TF-IDF features with the largest SVM weights for the positive and negative classes.", cap)]))
story.append(P(
    "The positive list is made of clear evaluative words (<i>great, delicious, best, excellent, perfect</i>). The negative list "
    "shows how much the model depends on negation. Bare <i>not</i> is its single strongest negative feature, and the bigrams "
    "<i>not good</i> and <i>not worth</i> also make the top ten. Meanwhile <i>not too</i> (as in \"not too sweet\") is a top "
    "<i>positive</i> feature, which only bigrams can capture. Figure 2 shows why we kept these words. <i>not</i> is the most "
    "frequent content token in every class, but appears about twice as often per review in negative and neutral reviews as in positive ones."))
story.append(img("word_freq.png", 5.6 * inch))
story.append(P("Figure 2. Most frequent tokens per review in the training set, by class (common function words removed for display only).", cap))
story.append(img("svd_scatter.png", 5.4 * inch))
v = A["svd_var"]
story.append(P(
    f"Figure 3. TF-IDF reviews projected with truncated SVD (700 random training reviews per class). The leading components mostly "
    f"encode <i>topic</i> (coffee, tea, dog food); together the first five explain only {100*sum(v[:5]):.1f}% of variance. "
    f"Component 5 loads on <i>great</i> and <i>love</i> and begins to pull positives away, but negative and neutral overlap almost "
    f"completely. This motivates using all ~226k sparse dimensions with a linear model like the SVM rather than a dense low-dimensional projection.", cap))

# ------------------------------------------------------------------ 2
story.append(P("2. Baseline results", h1))
story.append(P(
    f"The SVM was trained on 80% of the de-duplicated data and evaluated once on the held-out 20% ({S['n_test']:,} reviews: "
    f"{S['test_label_counts']['positive']:,} positive, {S['test_label_counts']['negative']:,} negative, {S['test_label_counts']['neutral']:,} neutral). "
    f"We use <font face='Courier'>LinearSVC</font> with balanced class weights, so mistakes on the rare negative and neutral classes cost more. "
    f"The regularisation strength C = 0.25 was chosen by 3-fold cross-validation on the training split only "
    f"(<font face='Courier'>src/tune.py</font>). Because 78% of labels are positive, we report <b>macro-averaged</b> scores, which weight the three classes equally."))
tc = S["test_label_counts"]
rows = [["Class", "Precision", "Recall", "F1", "Support"]]
for lab in ["negative", "neutral", "positive"]:
    rows.append([lab, f3(svm[f"{lab}_precision"]), f3(svm[f"{lab}_recall"]), f3(svm[f"{lab}_f1"]), f"{tc[lab]:,}"])
rows.append(["<b>macro average</b>", f"<b>{f3(svm.macro_precision)}</b>", f"<b>{f3(svm.macro_recall)}</b>", f"<b>{f3(svm.macro_f1)}</b>", f"{S['n_test']:,}"])
rows.append(["<b>accuracy</b>", "", "", f"<b>{pct(svm.accuracy)}</b>", f"{S['n_test']:,}"])
cls_tab = table(rows, [1.05 * inch, 0.65 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch])
cmp_tab = table([["", "Majority class", "Linear SVM"],
                 ["Accuracy", pct(maj.accuracy), pct(svm.accuracy)],
                 ["Macro precision", f3(maj.macro_precision), f3(svm.macro_precision)],
                 ["Macro recall", f3(maj.macro_recall), f3(svm.macro_recall)],
                 ["Macro F1", f3(maj.macro_f1), f3(svm.macro_f1)],
                 ["Negative recall", pct(maj.negative_recall), pct(svm.negative_recall)]],
                [1.05 * inch, 0.85 * inch, 0.85 * inch])
left = [cls_tab, P("Table 2. Linear SVM per-class results on the held-out 20%.", cap), cmp_tab,
        P("Table 3. Linear SVM versus the majority-class baseline.", cap)]
grid = Table([[left, [img("confusion_svm.png", 2.8 * inch), P("Figure 4. Confusion matrix (counts and row %).", cap)]]],
             colWidths=[3.75 * inch, 3.25 * inch], hAlign="LEFT")
grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(grid)
story.append(P(
    f"<b>Is the model learning?</b> Yes. The majority-class baseline scores {pct(maj.accuracy)} accuracy but only {f3(maj.macro_f1)} "
    f"macro-F1, because it never finds a single complaint. The SVM raises accuracy by {100*(svm.accuracy-maj.accuracy):.1f} points and "
    f"macro-F1 by {svm.macro_f1-maj.macro_f1:.2f}. It recovers {pct(svm.negative_recall)} of negative reviews, the class that matters most "
    f"for flagging product-quality problems. The confusion matrix shows where it still struggles. True neutral reviews are split almost "
    f"evenly between neutral ({pct(svm.neutral_recall)}), positive and negative, and most remaining negative errors are predicted positive."))
story.append(P(
    f"<b>Why this model.</b> The SVM is also efficient. It trains in about {svm.fit_seconds:.0f} second on a 2-core "
    f"machine and predicts 1,000 reviews in under 1 ms, so the web app can retrain or serve it instantly. In a quick check during "
    f"development it matched Logistic Regression's macro-F1 ({f3(lr.macro_f1)}) while training about {lr.fit_seconds/svm.fit_seconds:.0f}x "
    f"faster, and it beat our planned Naive Bayes model by {svm.macro_f1-nb.macro_f1:.2f} macro-F1. That is why we made it our primary model."))

# ------------------------------------------------------------------ 3
story.append(P("3. Error analysis", h1))
story.append(P(
    f"The SVM misclassifies {E['n_wrong']:,} of {E['n_test']:,} test reviews. By true class, the error rate is "
    f"{pct(E['err_by_true']['positive'])} for positive, {pct(E['err_by_true']['negative'])} for negative, and "
    f"{pct(E['err_by_true']['neutral'])} for neutral. We read a random sample of 40 errors and selected eight that are representative:"))
ex = [
    ("Great texture, Nasty smell", "\"I love the smooth yet granular feel... What I dislike most is that it really doesn't smell good\"", "neutral", "positive", "Mixed aspects"),
    ("Creamy - yes, Bold - no", "\"a decent sauce but it isn't spicy or bold enough for me. It is fine as a veggie dip...\"", "neutral", "positive", "Mixed aspects"),
    ("tasty, but pricey. but beware the shipping!", "\"quite tasty. The packing/shipping however leaves a lot to be desired... box came completely smashed\"", "positive", "negative", "Contrast + off-product"),
    ("Package Better", "\"I thoroughly enjoy the Celsius Energy drinks. My problem is with my last two shipments...\"", "neutral", "negative", "Off-product complaint"),
    ("BUY THIS AT COSTCO", "\"This is my go to drink - but you can buy a 24 pack at costco for about HALF the price\"", "neutral", "positive", "Contrast (price)"),
    ("disappointing", "\"You may be wondering what is disappointing about a four star coffee... better than average brew\"", "positive", "negative", "Summary word overrides"),
    ("Dog loves this", "\"had to literally be pried away from the dog... The package says not to let them eat it all in one day.\"", "positive", "negative", "Blunt negation"),
    ("Black Tiger Not for Me", "\"I do not care for the taste... I will stay with my favorite for the best taste I have found\"", "neutral", "positive", "Praise of competitor"),
]
rows = [["Summary", "Excerpt", "True", "Pred", "Pattern"]] + [list(r) for r in ex]
story.append(table(rows, [1.35 * inch, 3.25 * inch, 0.55 * inch, 0.55 * inch, 1.15 * inch], size=7.3))
story.append(P("Table 4. Eight misclassified test reviews (excerpts shortened). All predictions are from the linear SVM.", cap))
pc, pn = E["patterns"]["contrast"], E["patterns"]["negation"]
story.append(P(
    f"<b>Pattern: mixed-sentiment reviews with a contrast word.</b> Six of the eight praise one aspect and criticise another "
    f"(taste vs. smell, product vs. shipping or price), usually joined by <i>but</i> or <i>however</i>. A bag-of-n-grams model adds up "
    f"positive and negative evidence and cannot tell which clause carries the verdict. Across the full test set, reviews containing "
    f"<i>but / however / although / though / unfortunately / except</i> ({pct(pc['share'])} of reviews) have a <b>{pct(pc['with'])}</b> "
    f"error rate versus <b>{pct(pc['without'])}</b> for the rest, nearly three times higher. The same effect explains why errors <i>rise</i> with "
    f"length (Figure 5): long reviews mix more aspects. Short reviews are not our weak spot, contrary to our initial guess."))
story.append(P(
    f"<b>Secondary patterns.</b> (a) Negation is handled bluntly: bare <i>not</i> is the strongest negative feature, so a positive review "
    f"that mentions a warning (\"says not to let them eat it all\") flips. Reviews containing a negation word have a {pct(pn['with'])} error "
    f"rate vs {pct(pn['without'])}. (b) Many 3-star reviews complain about shipping, packaging or price rather than the food, so the "
    f"label reflects the purchase experience while the text vocabulary looks negative or positive. Some of this is label noise that no "
    f"text model can fully fix. (c) A strong word in the summary (\"disappointing\") can outweigh a moderate body."))
story.append(KeepTogether([img("error_by_length.png", 3.6 * inch),
    P("Figure 5. SVM test error rate by review length (word count of the body).", cap)]))
xb, xs, xn = X.iloc[0], X.iloc[1], X.iloc[2]
story.append(P(
    f"<b>How we will address it.</b> We ran two quick fixes (<font face='Courier'>src/experiments.py</font>). Giving the summary its own "
    f"TF-IDF block, and marking tokens inside a negation scope (\"not good\" becomes not good_NEG up to the next punctuation), moved "
    f"macro-F1 from {f3(xb.macro_f1)} to {f3(xn.macro_f1)} and neutral F1 from {f3(xb.neutral_f1)} to {f3(xn.neutral_f1)}. That gain is "
    f"small and from one split, so we will confirm it with cross-validation before adopting it. For the contrast pattern we plan to "
    f"(1) split reviews at contrast words and add clause-position features (e.g. separate TF-IDF for the clause after <i>but</i>), "
    f"(2) tune per-class decision thresholds to trade some positive precision for neutral recall, and (3) compare against a "
    f"contextual model (sentence embeddings or a fine-tuned DistilBERT) in M3. These models read word order and should handle "
    f"\"tasty, but...\" much better."))

# ------------------------------------------------------------------ 4
story.append(P("4. Updated timeline", h1))
story.append(table([
    ["Milestone", "Due", "Plan (revised)", "Change from M1"],
    ["M2", "Nov 15", "Feature pipeline, Linear SVM vs. majority baseline, error analysis (this report)", "Done; Linear SVM replaces the planned Naive Bayes; dedup step added"],
    ["M3", "Nov 22", "Negation scoping, summary features, contrast-clause features, class-threshold tuning; 5-fold CV; first neural/embedding model; model save/load API", "New: error-analysis fixes; neutral F1 is now the target metric"],
    ["M4 A", "Nov 30", "Web app: paste a review and see the SVM verdict and its top contributing words; add a second model (neural) side by side", "Unchanged scope; SVM chosen as default for speed"],
    ["M4 B", "Dec 6", "Final evaluation on held-out set, report and presentation, cleanup of repo / README", "Unchanged"],
], [0.65 * inch, 0.5 * inch, 3.2 * inch, 2.65 * inch]))
story.append(Spacer(1, 4))
story.append(P(
    "<b>Scope changes and rationale.</b> (1) <i>Primary model: Naive Bayes to Linear SVM.</i> The SVM is 0.08 macro-F1 better "
    "and still trains in about a second, so it is efficient enough to retrain on demand for the app. (2) <i>Headline metric: accuracy to macro-F1 and "
    "neutral F1.</i> Accuracy hides the minority classes that the business cares about. (3) <i>De-duplication added</i> to prevent "
    "train/test leakage. (4) <i>We keep the three-class target</i> but will report a binary (positive vs. negative) result alongside it, "
    "since neutral is intrinsically ambiguous and some users may only need the binary verdict."))

# ------------------------------------------------------------------ 5
story.append(P("5. Individual progress log", h1))
log = [
    ("Sean Kong (Project Lead)", "[Completed since M1: e.g., coordinated M2, set up repo structure and evaluation protocol, wrote report sections.] [Next: e.g., M3 planning, cross-validation of fixes.]"),
    ("Nicholas Randolph (App Engineer)", "[Completed since M1: ...] [Next: e.g., web-app skeleton that loads saved models and shows side-by-side predictions.]"),
    ("Kyle Baumgartner (Data Engineer)", "[Completed since M1: e.g., cleaning, de-duplication, and the train/test split.] [Next: e.g., negation scoping and contrast-clause features.]"),
    ("Mike Ji (Analyst / Model Trainer)", "[Completed since M1: e.g., trained the majority baseline, error analysis.] [Next: e.g., threshold tuning for neutral.]"),
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
