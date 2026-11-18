# alter the path to save the sample

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", type=Path, help="Path to the downloaded Reviews.csv")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--size", type=int, default=50000)
    args = parser.parse_args()
    output = ROOT / "data" / f"group_sample_seed{args.seed}.csv"
    if output.exists():
        parser.error(f"The sample already exists: {output}")
    if args.size <= 0:
        parser.error("Sample size must be positive.")

    reviews = pd.read_csv(args.reviews)
    if len(reviews) < args.size:
        parser.error(f"Requested {args.size:,} rows, but the source has {len(reviews):,}.")
    sample = reviews.sample(n=args.size, random_state=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    print(f"Saved {len(sample):,} rows to {output}")


if __name__ == "__main__":
    main()
