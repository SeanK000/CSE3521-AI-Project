# Saved models

Run `python src/pipeline.py` from the project folder to create
`models/artifacts.joblib`. This file contains the fitted vectorizers and models,
plus the training features used by the analysis scripts.

Run `python src/distilbert_model.py` (after installing
`requirements-model.txt`) to create `models/distilbert/` with the fine-tuned
DistilBERT weights, tokenizer, and run metadata.

Serialized models are generated locally and ignored by Git. Retrain after
changing the data, preprocessing, or dependency versions.
