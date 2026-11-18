# Saved models

Run `python src/pipeline.py` from the project folder to create
`models/artifacts.joblib`. This file contains the fitted vectorizers and models,
plus the training features used by the analysis scripts.

Serialized models are generated locally and ignored by Git. Retrain after
changing the data, preprocessing, or dependency versions.
