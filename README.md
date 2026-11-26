# CSE3521-AI-Project — Amazon Fine Food review sentiment

Group Phi: Sean Kong, Nicholas Randolph, Kyle Baumgartner, Mike Ji, and Ian Roy.

Three-class sentiment (negative 1-2★ / neutral 3★ / positive 4-5★) from review
summary and text, on the shared 50k sample `data/group_sample_seed47.csv` (seed 47).
Cleaning and de-duplication leave 46,315 reviews, split into 37,052 training
reviews and 9,263 held-out test reviews.

## Run (Milestone 2)

Run from the repository root with the shared sample in `data/`. See
[data/README.md](data/README.md) for download instructions. If the sample is
missing, create it with `python sample.py path/to/Reviews.csv`.

```bash
python -m pip install -r requirements.txt
python src/pipeline.py      # clean + dedup, BoW & TF-IDF, 4 classifiers + majority baseline
python src/analysis.py      # top features + figures -> figures/, error analysis -> results/
python src/experiments.py   # training-only validation of summary/negation features
python src/tune.py          # (optional) training-only 3-fold CV comparison of C for LR and SVM
python src/build_report.py  # Milestone2_Report.pdf in the repository root
```

The pipeline saves fitted vectorizers and models to `models/artifacts.joblib`.
Both feature vocabularies are fitted on training reviews only. Tuning refits
TF-IDF within each fold; tuning and experiment scores use training-only
validation data and are separate from the test results below. These scripts
do not automatically change the initial model settings.

To open the local Streamlit app after training:

```bash
python -m streamlit run src/streamlitApplication.py
```

## Milestone 2 results (held-out 20%, 9,263 reviews after dedup)

| Model | Features | Accuracy | Macro F1 | Train time |
|---|---|---|---|---|
| Majority class | – | 77.74% | 0.292 | 0.01 s |
| Multinomial NB | BoW | 83.82% | 0.651 | 0.05 s |
| Complement NB | TF-IDF | 85.62% | 0.559 | 0.07 s |
| Logistic Regression | TF-IDF 1-2 gram | 88.17% | 0.725 | 4.83 s |
| **Linear SVM** | TF-IDF 1-2 gram | **89.15%** | **0.722** | **0.66 s** |

Linear SVM is the most accurate M2 model and trains faster than Logistic
Regression in this run. Logistic Regression has slightly higher macro F1.
Training times cover classifier fitting only and vary by machine.

Full metrics, confusion matrices, predictions, and error examples are described
in [results/README.md](results/README.md).

## Milestone 3 data engineering

M3 preserves the M2
sample and splits, packages summary/body and negation features, and adds a
contrast-clause candidate. Reproduce the data artifacts from the repository root:

```bash
python -m pip install -r requirements-data.txt
python src/prepare_m3_data.py
python -m unittest discover -s tests -v
python src/verify_m3_data.py
```

Use Python 3.12 in a virtual environment and the unchanged shared sample. Large
generated data and feature artifacts stay local in ignored `data/m3/`; compact
audits are in `results/m3_data/`. Candidate vocabularies are fitted on inner
training only. The shared sample checksum accepts LF and CRLF line endings.

### Train and run M3

From the repository root, with the shared sample in `data/` and the dependencies
from `requirements.txt` installed:

```bash
python src/train_m3.py
python -m streamlit run src/streamlitApplication.py
```

The group compares four feature variants with Linear SVM and Logistic
Regression using validation macro F1. The feedforward model uses selected
TF-IDF features and two ReLU hidden layers (64 and 32 units). It selects its
training length by validation macro F1. Both selected models are fitted on the
37,052 outer training reviews and evaluated on the original 9,263 test reviews.
The trainer uses prepared splits when available and recreates them from the
locked sample otherwise.

The app loads the two M3 models from `models/m3_models.joblib`. If the M2
bundle is available, it also includes the classical baselines.

- [Unified test comparison](results/m3/model_comparison.csv) and
  [confusion matrices](results/m3/confusion_matrices.json)
- [Classical validation comparison](results/m3_classical/validation_comparison.csv)
  and [test metrics](results/m3_classical/test_metrics.json)
- [Neural test metrics](results/m3_neural/test_metrics.json),
  [epoch history](results/m3_neural/history.csv), and
  [training and validation curves](figures/m3_neural_loss.png)

### Optional DistilBERT experiment

DistilBERT can be trained separately with its optional PyTorch stack:

```bash
python -m pip install -r requirements-model.txt
python src/distilbert_model.py
```

It fine-tunes `distilbert-base-uncased` with DistilBERT's `(Summary, Text)`
sentence-pair input, selects the epoch on the training-only validation split,
refits on all outer training reviews, and scores the held-out test split once.
The model is saved to the ignored `models/distilbert/`; metrics, predictions,
loss history, and the loss curve go to `results/m3_distilbert/` and
`figures/distilbert_loss.png`. See
[docs/M3_DISTILBERT.md](docs/M3_DISTILBERT.md) for the protocol and options.
