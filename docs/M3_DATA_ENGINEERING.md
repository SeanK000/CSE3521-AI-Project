# M3 Data Engineering

I kept the same train and test split we used in M2 so we could compare models on the same reviews. I reused our negation code and added a feature that separates the two sides of a phrase like “tasty but expensive.” I saved all four versions of the features for the team to try and checked that none of the test reviews showed up in the training data.

The modern-model owner can fine-tune DistilBERT on the raw `Summary`/`Text` fields in these splits with `src/distilbert_model.py`; see `docs/M3_DISTILBERT.md` for the protocol and outputs.
