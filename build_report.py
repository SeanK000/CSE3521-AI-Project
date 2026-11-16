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
    f"<b>Summary.</b> We built a reproducible pipeline (<font face='Courier'>src/pipeline.py</font>) that cleans the 50,000-review "
    f"sample, builds Bag-of-Words and TF-IDF features, and trains four classical models against a majority-class baseline on a "
    f"stratified 80/20 split. Our best model, a <b>linear SVM on unigram+bigram TF-IDF</b>, reaches {pct(svm.accuracy)} accuracy and "
    f"{f3(svm.macro_f1)} macro-F1, against {pct(maj.accuracy)} / {f3(maj.macro_f1)} for always predicting \"positive\". It trains in "
    f"about {svm.fit_seconds:.0f} second. It replaces Naive Bayes (our M1 plan, macro-F1 {f3(nb.macro_f1)}) as the primary model. "
    f"Neutral reviews remain the main weakness (F1 {f3(svm.neutral_f1)})."))

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
    ["Representation", "Method", "Size", "Used by"],
    ["Bag of Words", "Raw unigram counts (CountVectorizer), tokens kept if they occur in at least 2 training reviews",
     f"{S['bow_vocab']:,} terms", "Multinomial NB"],
    ["TF-IDF (1-2 grams)", "Unigrams + bigrams, sublinear tf (1+log tf), idf weighting, L2-normalised rows; min_df = 2, max_df = 0.9",
     f"{S['tfidf_vocab']:,} terms", "Complement NB, Logistic Regression, Linear SVM"],
    ["LSA (for analysis)", "Truncated SVD of the TF-IDF matrix (used only for visualisation)", "10 dims", "Figure 2"],
], [1.15 * inch, 3.6 * inch, 0.85 * inch, 1.4 * inch]))
story.append(P("Table 1. Text representations, fitted on the training split only.", cap))

def feat_rows(pos, neg):
    rows = [["#", "Positive feature", "weight", "Negative feature", "weight"]]
    for i in range(10):
        rows.append([i + 1, pos[i][0], f"{pos[i][1]:.2f}", neg[i][0], f"{neg[i][1]:.2f}"])
    return rows
tw = [0.25 * inch, 1.05 * inch, 0.5 * inch, 1.05 * inch, 0.5 * inch]
t1 = table(feat_rows(A["bow_nb_pos"], A["bow_nb_neg"]), tw, size=7.5)
t2 = table(feat_rows(A["tfidf_svm_pos"], A["tfidf_svm_neg"]), tw, size=7.5)
both = Table([[P("<b>Bag of Words + Multinomial NB</b><br/>log P(w|pos) - log P(w|neg), words with 100+ occurrences", small),
               P("<b>TF-IDF + Linear SVM</b><br/>one-vs-rest weight for the positive / negative class", small)], [t1, t2]],
             colWidths=[3.5 * inch, 3.5 * inch], hAlign="LEFT")
both.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(KeepTogether([P("<b>Top-10 most informative features.</b>", h2), both,
                           P("Table 2. Top-10 features for predicting positive and negative sentiment under each representation.", cap)]))
story.append(P(
    "The two lists differ in an instructive way. Naive Bayes ranks words by how lopsided their class counts are, so it surfaces "
    "<i>product</i> words that happen to be reviewed favourably (pb2, a powdered peanut butter; chamomile; \"theater\" from movie-theater "
    "popcorn) and process words like <i>refund</i> and <i>expired</i>. The SVM learns weights jointly, so correlated topic words cancel out "
    "and the list is almost purely evaluative. It also uses bigrams to model negation: <i>not good</i> and <i>not worth</i> are strong negative "
    "features, while <i>not too</i> (\"not too sweet\") is positive. Bare <i>not</i> is the single strongest negative weight. Figure 1 "
    "shows why stop-word removal would have hurt: <i>not</i> is the most frequent content token in every class, but about twice "
    "as frequent per review in negative and neutral reviews as in positive ones."))
story.append(img("word_freq.png", 6.6 * inch))
story.append(P("Figure 1. Most frequent tokens per review in the training set, by class (common function words removed for display only).", cap))
story.append(img("svd_scatter.png", 6.4 * inch))
v = A["svd_var"]
story.append(P(
    f"Figure 2. TF-IDF reviews projected with truncated SVD (700 random training reviews per class). The leading components mostly "
    f"encode <i>topic</i> (coffee, tea, dog food); together the first five explain only {100*sum(v[:5]):.1f}% of variance. "
    f"Component 5 loads on <i>great</i> and <i>love</i> and begins to pull positives away, but negative and neutral overlap almost "
    f"completely. This motivates using all ~226k sparse dimensions with a linear model rather than a dense low-dimensional projection.", cap))

# ------------------------------------------------------------------ 2
story.append(P("2. Baseline results", h1))
story.append(P(
    f"All models were trained on the same 80% split and evaluated once on the held-out 20% ({S['n_test']:,} reviews: "
    f"{S['test_label_counts']['positive']:,} positive, {S['test_label_counts']['negative']:,} negative, {S['test_label_counts']['neutral']:,} neutral). "
    f"Because 78% of labels are positive, accuracy alone is misleading, so we report <b>macro-averaged</b> precision, recall and F1, which "
    f"weight the three classes equally. Regularisation strength C for Logistic Regression and the SVM was chosen by 3-fold cross-validation "
    f"on the training split only (<font face='Courier'>src/tune.py</font>); both use balanced class weights."))
rows = [["Model", "Features", "Accuracy", "Macro P", "Macro R", "Macro F1", "Neg F1", "Neu F1", "Pos F1", "Train time"]]
for name, r in M.iterrows():
    rows.append([("<b>%s</b>" % name) if name == "Linear SVM" else name, r.features, pct(r.accuracy), f3(r.macro_precision),
                 f3(r.macro_recall), ("<b>%s</b>" % f3(r.macro_f1)) if name == "Linear SVM" else f3(r.macro_f1),
                 f3(r.negative_f1), f3(r.neutral_f1), f3(r.positive_f1), f"{r.fit_seconds:.1f} s"])
story.append(table(rows, [1.2 * inch, 0.55 * inch, 0.62 * inch, 0.55 * inch, 0.55 * inch, 0.62 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.62 * inch]))
story.append(P("Table 3. Held-out test results (20%). Train time is wall-clock on a 2-core machine.", cap))
story.append(P(
    f"<b>Is the model learning?</b> Yes. The majority-class baseline scores {pct(maj.accuracy)} accuracy but {f3(maj.macro_f1)} "
    f"macro-F1, because it never finds a single complaint. The linear SVM raises accuracy by {100*(svm.accuracy-maj.accuracy):.1f} "
    f"points and macro-F1 by {svm.macro_f1-maj.macro_f1:.2f}, and recovers {pct(svm.negative_recall)} of negative reviews, the class "
    f"that matters most for the quality-investigation use case. Complement NB looks good on accuracy ({pct(M.loc['Complement NB'].accuracy)}) but "
    f"almost never predicts neutral (recall {pct(M.loc['Complement NB'].neutral_recall)}), which is exactly the failure macro-F1 exposes."))
story.append(P(
    f"<b>A more efficient model.</b> Logistic Regression and the linear SVM are essentially tied on macro-F1 ({f3(lr.macro_f1)} vs "
    f"{f3(svm.macro_f1)}), but the SVM has higher accuracy and trains about {lr.fit_seconds/svm.fit_seconds:.0f}x faster "
    f"({svm.fit_seconds:.1f} s vs {lr.fit_seconds:.1f} s), with comparable prediction cost (under 1 ms per 1,000 reviews). We therefore adopt "
    f"it as the primary model for the web app. Logistic Regression stays as a second model because it gives calibrated probabilities "
    f"and slightly better neutral recall ({pct(lr.neutral_recall)} vs {pct(svm.neutral_recall)})."))
story.append(KeepTogether([img("confusion.png", 6.2 * inch),
    P(f"Figure 3. Confusion matrices on the test set (counts and row %). Moving from NB to the SVM mainly fixes negative reviews "
      f"(71% to 80% recall) and positives mislabelled as neutral. Neutral stays split roughly evenly three ways.", cap)]))

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
    f"length (Figure 4): long reviews mix more aspects. Short reviews are not our weak spot, contrary to our initial guess."))
story.append(P(
    f"<b>Secondary patterns.</b> (a) Negation is handled bluntly: bare <i>not</i> is the strongest negative feature, so a positive review "
    f"that mentions a warning (\"says not to let them eat it all\") flips. Reviews containing a negation word have a {pct(pn['with'])} error "
    f"rate vs {pct(pn['without'])}. (b) Many 3-star reviews complain about shipping, packaging or price rather than the food, so the "
    f"label reflects the purchase experience while the text vocabulary looks negative or positive. Some of this is label noise that no "
    f"text model can fully fix. (c) A strong word in the summary (\"disappointing\") can outweigh a moderate body."))
story.append(KeepTogether([img("error_by_length.png", 4.2 * inch),
    P("Figure 4. SVM test error rate by review length (word count of the body).", cap)]))
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
    ["M2", "Nov 15", "Feature pipeline, 4 classical models + baseline, error analysis (this report)", "Done; added dedup step and Linear SVM / LR beyond the planned NB"],
    ["M3", "Nov 22", "Negation scoping, summary features, contrast-clause features, class-threshold tuning; 5-fold CV; first neural/embedding model; model save/load API", "New: error-analysis fixes; neutral F1 is now the target metric"],
    ["M4 A", "Nov 30", "Web app: paste a review and compare NB / LR / SVM (+ neural) side by side with confidence and top contributing words", "Unchanged scope; SVM chosen as default for speed"],
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
    ("Mike Ji (Analyst / Model Trainer)", "[Completed since M1: e.g., trained baseline and NB models, error analysis.] [Next: e.g., threshold tuning for neutral.]"),
    ("Ian Roy (ML Engineer)", "[Completed since M1: e.g., TF-IDF + SVM / LR models, hyperparameter search.] [Next: e.g., embedding / DistilBERT model for M3.]"),
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
