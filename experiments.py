"""Quick follow-up experiments suggested by the error analysis (Linear SVM).
Run after pipeline.py:  python src/experiments.py"""
import re, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from pipeline import load, clean, SEED, RES

NEG = re.compile(r"\b(?:not|no|never|n't|cannot|isn't|wasn't|don't|doesn't|didn't)\b")
def neg_scope(t):
    """Append _NEG to the tokens after a negation word, up to the next punctuation."""
    out = []
    for clause in re.split(r"([.,!?;:])", t):
        toks, on = clause.split(), False
        for w in toks:
            out.append(w + "_NEG" if on else w)
            if NEG.search(w): on = True
    return " ".join(out)

df, _ = load()
df["summ"] = df["Summary"].map(clean); df["body"] = df["Text"].map(clean)
tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
def run(name, Xtr, Xte):
    m = LinearSVC(C=0.25, class_weight="balanced").fit(Xtr, tr.label); p = m.predict(Xte)
    r = dict(variant=name, accuracy=accuracy_score(te.label, p), macro_f1=f1_score(te.label, p, average="macro"),
             neutral_f1=f1_score(te.label, p, labels=["neutral"], average="macro"),
             n_features=Xtr.shape[1]); print(r); return r
V = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
rows = []
v = V(); rows.append(run("baseline: summary+text in one TF-IDF", v.fit_transform(tr.doc), v.transform(te.doc)))
vs, vb = V(), V()
rows.append(run("separate summary & body TF-IDF", hstack([vs.fit_transform(tr.summ), vb.fit_transform(tr.body)]).tocsr(),
                hstack([vs.transform(te.summ), vb.transform(te.body)]).tocsr()))
vs, vb = V(), V()
trb, teb = tr.body.map(neg_scope), te.body.map(neg_scope)
trs, tes = tr.summ.map(neg_scope), te.summ.map(neg_scope)
rows.append(run("separate + negation scoping", hstack([vs.fit_transform(trs), vb.fit_transform(trb)]).tocsr(),
                hstack([vs.transform(tes), vb.transform(teb)]).tocsr()))
pd.DataFrame(rows).to_csv(RES / "experiments.csv", index=False)
