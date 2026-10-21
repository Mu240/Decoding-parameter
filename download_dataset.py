
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download a sample of the TruthfulQA generation dataset."
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/truthfulqa_sample.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TruthfulQA from Hugging Face...")
    dataset = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
    )

    df = dataset.to_pandas()

    if args.limit > 0 and args.limit < len(df):
        df = df.sample(
            n=args.limit,
            random_state=args.seed,
        )

    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} questions to: {output_path}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()