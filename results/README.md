# Results files

This folder contains the saved evaluation and analysis outputs for Milestone 2.
The scripts generate these files; rerunning a script replaces its outputs.
Update the code and regenerate results rather than editing scores by hand.

## Files at a glance

| File | Created by | What it contains |
| --- | --- | --- |
| [model_comparison.csv](model_comparison.csv) | [pipeline.py](../src/pipeline.py) | Held-out test scores and timing for each classifier and the majority-class baseline. |
| [summary.json](summary.json) | [pipeline.py](../src/pipeline.py) | Dataset counts, data-quality checks, settings, vocabulary sizes, and confusion matrices. |
| [test_predictions.csv](test_predictions.csv) | [pipeline.py](../src/pipeline.py) | Every held-out review, its true label, and predictions from all models. |
| [analysis.json](analysis.json) | [analysis.py](../src/analysis.py) | Important features, word frequencies, SVD statistics, and SVM error statistics. |
| [misclassified_svm.csv](misclassified_svm.csv) | [analysis.py](../src/analysis.py) | All held-out reviews that the Linear SVM classified incorrectly. |
| [error_examples.csv](error_examples.csv) | [analysis.py](../src/analysis.py) | Eight reproducibly selected SVM mistakes for manual inspection. |
| [tuning.csv](tuning.csv) | [tune.py](../src/tune.py) | Training-only cross-validation scores for different model settings. |
| [experiments.csv](experiments.csv) | [experiments.py](../src/experiments.py) | Training-only validation scores for alternative text features. |

## Which scores belong together?

`model_comparison.csv` reports performance on the outer 20% held-out test split.
All its models use the same test reviews. Sentiment labels come from the star
rating: 1-2 stars are negative, 3 neutral, and 4-5 positive.

`tuning.csv` and `experiments.csv` use validation data drawn only from the outer
training split. They support decisions about settings and features. Their scores
are not directly comparable to the held-out test scores because the training
sizes and evaluation reviews differ. Neither script automatically changes the
model settings in `pipeline.py` or the model served by the app.

Scores and error rates use fractions from 0 to 1. For example, an accuracy of
`0.8915` means approximately 89.15% of predictions were correct. Feature weights
and timing values use their own units, described below.

## model_comparison.csv

Each row describes one model: Majority class, Multinomial NB, Complement NB,
Logistic Regression, or Linear SVM.

| Column | Meaning |
| --- | --- |
| `model` | Classifier name. The majority baseline always predicts the most frequent training label. |
| `features` | Input representation: `BoW`, `TF-IDF`, or `-` for the majority baseline. |
| `accuracy` | Fraction of test reviews classified correctly. |
| `macro_precision` | Average class precision, with equal weight for each class. Precision asks how often a predicted class is correct. |
| `macro_recall` | Average class recall, with equal weight for each class. Recall asks how many actual members of a class were found. |
| `macro_f1` | Average of the three class F1 scores. Each class F1 balances precision and recall. |
| `weighted_f1` | Average class F1 weighted by the number of true test reviews in each class. |
| `fit_seconds` | Classifier fitting time in seconds; excludes text cleaning and feature extraction. |
| `predict_ms_per_1k` | Classifier prediction time normalized to milliseconds per 1,000 reviews; excludes feature extraction. |
| `negative_*`, `neutral_*`, `positive_*` | Precision, recall, and F1 for the named class. |

Most reviews are positive, so accuracy and weighted F1 can hide weak performance
on neutral or negative reviews. Read macro F1 and the individual class scores
alongside accuracy. Timing varies with the machine and other running work.

## summary.json

This records the context needed to interpret the evaluation:

- `n_raw`, `n_dedup`, `n_train`, `n_test`: input sample, retained reviews after
  validation/deduplication, training rows, and held-out test rows.
- `seed`, `test_fraction`, `labels`: random seed, held-out fraction, and class order.
- `data_quality`: missing-value counts, rejected rows, duplicate counts, and the
  number of duplicate-body groups with conflicting sentiment labels.
  `missing_text` and `missing_summary` count missing values in the raw sample.
  `duplicates_removed` is the total of `exact_duplicates_removed` and the
  additional `cleaning_equivalent_duplicates_removed`; do not add all three.
  The first occurrence of a duplicate cleaned body is retained.
- `model_settings`, `feature_settings`: classifier parameters, the stated source
  of those settings, and vectorizer parameters used for this run.
- `label_counts`, `test_label_counts`: class counts in all retained reviews and
  in the test split, respectively.
- `bow_vocab`, `tfidf_vocab`: numbers of features learned from training reviews.
- `mean_words`, `median_words`: cleaned review-body word lengths across retained
  reviews; the summary is excluded from these length measures.
- `confusion`: one confusion matrix per model. Rows are true labels and columns
  are predicted labels, both in the order given by `labels`:
  negative, neutral, positive.

For example, the matrix entry at row `negative`, column `positive` counts actual
negative reviews predicted as positive. Matrix entries are counts, not rates.

## test_predictions.csv

Each row is one held-out review. `Id` is its original dataset ID, `Score` is the
star rating, and `label` is the true sentiment derived from that rating.
`Summary` and `Text` retain the review content; `n_words` counts cleaned body
words. The model input combines the summary and body, then applies text cleaning.

Columns beginning with `pred_` contain each model's predicted sentiment, such as
`pred_Linear_SVM`. Compare one of these columns with `label` to find mistakes.
The star rating is used to establish the true label, not as a prediction input.

## analysis.json

The analysis script also creates the corresponding charts in [figures/](../figures/).

| Field | Meaning |
| --- | --- |
| `tfidf_svm_pos`, `tfidf_svm_neg` | Top positive and negative TF-IDF features, stored as `[feature, weight]` pairs. Weights are the corresponding SVM class coefficients. |
| `bow_nb_pos`, `bow_nb_neg` | Top BoW features ranked by the NB log-probability ratio for positive versus negative, or negative versus positive. These are not probabilities. |
| `bow_min_train_occurrences` | Minimum total training token occurrences required for a term to appear in the BoW importance display. This display filter does not change the trained model. |
| `word_freq` | Common training tokens from the combined summary/body text by class, stored as `[token, occurrences_per_review]`. Repeated tokens can make a value greater than 1. Common function words are excluded only from this display. |
| `svd_var` | Explained-variance ratios for the truncated-SVD components fitted to training TF-IDF features. |
| `svd_top_terms` | Terms with the largest absolute weights in each SVD component. Component keys start at `0`; plot labels start at component 1. |
| `svd_class_means` | Mean training-review coordinates for each class on the first components. |
| `errors` | Linear SVM error counts and rates, detailed below. |

The `errors` object includes:

- `n_wrong`, `n_test`: number of SVM mistakes and total test reviews.
- `by_len`: each review-body length bucket mapped to `[review_count, error_rate]`.
  Buckets are `<30`, `30-59`, `60-119`, and `120+`; the last has no upper limit.
- `by_len_nonpos`, `by_len_nonpos_n`: error rates and review counts for those
  buckets restricted to true negative/neutral reviews.
- `patterns`: error rates for reviews with and without contrast or negation
  markers. `with_n`/`without_n` are review counts;
  `with_wrong`/`without_wrong` are mistake counts; `share` is the fraction of
  all test reviews containing the marker.
- `err_by_true`, `n_by_true`, `wrong_by_true`: error rate, test count, and mistake
  count for each true class.
- `pair_counts`: mistake counts by direction, such as `neutral->positive`.

An empty subgroup has count 0 and rate `null`, rather than an error rate of zero.
Marker groups overlap, and their rates are descriptive associations; they do
not establish why a model made a particular mistake.

## misclassified_svm.csv and error_examples.csv

`misclassified_svm.csv` contains the rows where `pred_Linear_SVM` differs from
`label`. It preserves the columns from `test_predictions.csv` and adds:

- `pred`: the SVM prediction, repeated for convenient inspection.
- `err`: the error indicator, which is 1 for every row in this file.
- `contrast`, `negation`: whether the review body contains the corresponding
  text marker.

`error_examples.csv` is a smaller selection with columns `Id`, `Summary`,
`Text`, `true`, and `pred`. The script selects one example from each available
confusion pair, then fills the remaining slots from other errors, using the
project seed. It selects up to eight rows if enough mistakes exist.

Read the full summary and body before assigning an explanation. These are
automatically selected examples for the team's analysis, not manually verified
reasons for the errors.

## tuning.csv

Each row is one model and regularization setting `C`, evaluated with three-fold
cross-validation on the outer training split. Each fold fits its own TF-IDF
vocabulary and IDF weights.

- `n_training_reviews`: size of the outer training pool before the CV folds.
- `fold_1_macro_f1` through `fold_3_macro_f1`: validation macro F1 for each fold.
- `mean_macro_f1`, `std_macro_f1`: mean and population standard deviation of
  those three scores. The standard deviation is not a confidence interval.
- `mean_fit_seconds`: average fold fitting time, including TF-IDF fitting and
  transformation as well as classifier fitting. It has a different scope from
  `fit_seconds` in `model_comparison.csv`.
- `best_for_model`: marks the highest mean macro F1 among the tested settings
  for that model. It does not select the best classifier across different models.

## experiments.csv

Each row describes one feature variant: combined summary/body TF-IDF, separate
summary/body TF-IDF, or separate features with negation scoping.

- `evaluation_split`: identifies the scores as `training-only validation`.
- `n_train`, `n_validation`: sizes of the inner training and validation subsets.
  The validation subset is 20% of the outer training pool.
- `accuracy`, `macro_f1`, `neutral_f1`: scores on that validation subset.
- `n_features`: total feature count for that variant; separate summary/body
  representations have separate vocabularies.

## Regenerate the files

Run from the repository root after installing the dependencies and placing the
sample at `data/group_sample_seed47.csv`:

```bash
python src/pipeline.py
python src/analysis.py
python src/tune.py
python src/experiments.py
```

The last two commands are optional comparisons. Run the pipeline before analysis
so that `summary.json`, predictions, and `models/artifacts.joblib` all describe
the same training run. After changing the data, cleaning, split, or features,
regenerate the related outputs together to avoid mixing incompatible results.
