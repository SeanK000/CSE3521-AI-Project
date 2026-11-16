# CSE3521-AI-Project — Amazon Fine Food review sentiment

Three-class sentiment (negative 1-2★ / neutral 3★ / positive 4-5★) from review text,
on the shared 50k sample `group_sample_seed47.csv` (seed 47).

## Run (Milestone 2)
```bash
pip install -r requirements.txt
python src/pipeline.py      # clean + dedup, BoW & TF-IDF, train 5 models, metrics -> results/
python src/analysis.py      # top features, figures -> figures/, error stats -> results/analysis.json
python src/experiments.py   # negation-scoping / separate-summary experiments -> results/experiments.csv
python src/tune.py          # (optional) 3-fold CV used to pick C for LR and SVM
python src/build_report.py  # Milestone2_Report.pdf
```

## Milestone 2 results (held-out 20%, 9,266 reviews after dedup)
| Model | Features | Accuracy | Macro F1 | Train time |
|---|---|---|---|---|
| Majority class | – | 77.7% | 0.292 | 0 s |
| Multinomial NB | BoW | 83.9% | 0.651 | 0.1 s |
| Complement NB | TF-IDF | 85.5% | 0.556 | 0.1 s |
| Logistic Regression | TF-IDF 1-2 gram | 88.5% | 0.729 | ~14 s |
| **Linear SVM** | TF-IDF 1-2 gram | **89.4%** | **0.727** | **~1 s** |

Linear SVM is the primary model: same macro-F1 as LR, ~13x faster to train.
