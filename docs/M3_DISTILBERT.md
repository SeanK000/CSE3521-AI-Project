# DistilBERT sentiment model (M3)

`src/distilbert_model.py` fine-tunes `distilbert-base-uncased` for the same
three-class task used in Milestone 2. It keeps the locked seed-47 split so its
scores can be compared with the BoW/TF-IDF models on the same held-out reviews.

## Setup

Use Python 3.12 and install the optional model stack. It is separate from the
classical requirements because PyTorch is large and GPU installs are machine
specific.

```bash
python -m pip install -r requirements-model.txt
python src/prepare_m3_data.py
python src/distilbert_model.py
```

For a CUDA machine, install the matching PyTorch wheel from
[pytorch.org](https://pytorch.org/get-started/locally/) before installing the
rest of `requirements-model.txt`. The script uses CUDA when available, then
Apple MPS, then CPU. The first run downloads the pretrained checkpoint from
Hugging Face; pass a local directory to `--model-name` for an offline run.

The prepared `data/m3/` directory is preferred. If it is absent, the script
falls back to `data/group_sample_seed47.csv` and recreates the same outer and
inner splits with seed 47. It refuses to run if the ids or cleaned bodies in
the loaded splits overlap.

## Training protocol

* **Input:** DistilBERT's sentence-pair input, `(Summary, Text)`, using the M2
  cleaner (HTML/URL removal, whitespace normalization, lowercasing) and
  tokenized to at most 256 tokens by default. The summary and body keep separate
  token budgets instead of concatenating with the classical `.` separator.
* **Class imbalance:** cross-entropy uses balanced class weights by default.
  `--no-class-weights` disables this.
* **Validation:** the model trains on `inner_train` and early-stops on
  `validation` macro-F1. The best epoch is restored before reporting.
* **Final fit:** the model is then re-initialized and trained on all
  `outer_train` reviews for the selected number of epochs, matching the data
  used by the classical models. `--no-final-fit` skips this and scores the
  inner-training checkpoint instead.
* **Held-out test:** `test` is scored once after model selection. Use validation
  results for iteration; do not tune on the test split.
* **Reproducibility:** Python, NumPy, and PyTorch seeds are set from `--seed`
  (47 by default), and deterministic cuDNN settings are enabled.

Default settings are fixed rather than tuned on test: 4 maximum epochs,
batch size 16, learning rate `2e-5`, weight decay `0.01`, 10% linear warmup,
early-stopping patience 1, and maximum length 256. Run
`python src/distilbert_model.py --help` for every option.

## Outputs

The script writes only new M3 files; it does not change the M2 results or the
classical `models/artifacts.joblib`.

| Path | Contents |
| --- | --- |
| `models/distilbert/` | Final model, tokenizer, and `run_metadata.json`. |
| `results/m3_distilbert/history.csv` | Per-epoch training/validation loss and validation scores. |
| `results/m3_distilbert/validation_metrics.json` | Validation accuracy, macro/weighted F1, per-class scores, and confusion matrix. |
| `results/m3_distilbert/validation_predictions.csv` | Validation ids, labels, predictions, and class probabilities. |
| `results/m3_distilbert/test_metrics.json` | Held-out test scores for the final model. |
| `results/m3_distilbert/test_predictions.csv` | Held-out predictions and probabilities for error analysis. |
| `results/m3_distilbert/run_metadata.json` | Arguments, split counts, selected epoch, timings, package versions, and artifact paths. |
| `figures/distilbert_loss.png` | Training/validation loss and validation macro-F1 curves. |

The model directory is git-ignored because it is generated locally and is
large. The compact metrics and the figure are intended for the M3 report and
for comparison with `results/model_comparison.csv`.

## Comparison notes

DistilBERT and the classical models share the same test reviews and labels, but
they do not share the same input length. TF-IDF sees the full cleaned document;
DistilBERT truncates long reviews to `--max-length` tokens. Report that limit
alongside the test scores. The test predictions include class probabilities,
unlike the classical `predict` outputs, so error analysis should use the
predicted label column rather than the raw probabilities.

Do not copy test scores into `results/model_comparison.csv` by hand. Rerun the
script and cite `results/m3_distilbert/test_metrics.json`.
