from pathlib import Path
import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_csv", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    args = parser.parse_args()

    fold_csv = Path(args.fold_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fold_csv)
    slide_ids = []
    for col in ("train", "val", "test"):
        if col in df.columns:
            slide_ids.extend(df[col].dropna().astype(str).tolist())

    unique_ids = sorted(set(slide_ids))
    out_df = pd.DataFrame({"slide_id": unique_ids})
    out_df.to_csv(output_csv, index=False)

    print(f"Wrote {len(unique_ids)} slide IDs to: {output_csv}")


if __name__ == "__main__":
    main()
