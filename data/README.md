# Review data

Download the Amazon Fine Food Reviews dataset from
[Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and extract
`Reviews.csv` locally. Do not add the downloaded CSV to Git.

From the project folder, create the shared 50,000-review sample:

```bash
python sample.py path/to/Reviews.csv
```

This uses seed 47 and writes `data/group_sample_seed47.csv`. Share that same
sample within the team. The pipeline removes duplicate cleaned review bodies
before creating a stratified 80/20 train/test split, also with seed 47.

The original root-level sample has been moved here locally and removed from the
current checkout's root. It will still exist in old Git commits; this cleanup
does not rewrite repository history.

For M3, `python src/prepare_m3_data.py` creates ignored `data/m3/` artifacts:
raw/cleaned review records, ordered labels, split/CV manifests, a duplicate
audit, and four inner-training/validation feature variants. Do not commit this
directory or the source CSV. Use `python src/verify_m3_data.py` to check the
prepared artifacts. See [the handoff](../docs/M3_DATA_ENGINEERING.md) for the
data contract, setup commands, and loading examples.
