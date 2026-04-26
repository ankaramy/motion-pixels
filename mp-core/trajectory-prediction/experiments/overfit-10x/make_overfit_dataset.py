"""
make_overfit_dataset.py
-----------------------
Creates a 10x-duplicated version of trajectories_encoded.csv for the
overfitting experiment.  Each copy receives a unique person_id range so
IDs never collide across copies.

This is a non-destructive operation: the source CSV is never modified.

Logic adapted from duplicate_trajectories.py in the parent directory.

Usage
-----
  python make_overfit_dataset.py [--n-copies N]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE    = Path(__file__).resolve().parent
MP_ROOT = HERE.parent.parent.parent.parent   # motion-pixels/
MP_DATA = MP_ROOT / "mp-data"

SRC_CSV  = MP_DATA / "processed" / "encoded" / "trajectories_encoded.csv"
EXPR_OUT = MP_DATA / "outputs" / "prediction" / "experiments" / "overfit-10x"
OUT_CSV  = EXPR_OUT / "trajectories_overfit.csv"

DEFAULT_N_COPIES = 10


def duplicate(df: pd.DataFrame, id_col: str, n_copies: int) -> pd.DataFrame:
    max_id = int(df[id_col].max())
    stride = max_id + 1
    copies = []
    for i in range(n_copies):
        copy = df.copy()
        copy[id_col] = copy[id_col] + i * stride
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Create overfitting dataset")
    ap.add_argument("--n-copies", type=int, default=DEFAULT_N_COPIES,
                    help=f"Number of times to duplicate the dataset (default: {DEFAULT_N_COPIES})")
    args = ap.parse_args()

    if not SRC_CSV.exists():
        sys.exit(
            f"[ERROR] Source CSV not found: {SRC_CSV}\n"
            "        Run encode_space.py first to generate it."
        )

    df = pd.read_csv(SRC_CSV)
    rows_before = len(df)

    if "person_id" in df.columns:
        id_col = "person_id"
    elif "track_id" in df.columns:
        id_col = "track_id"
    else:
        sys.exit("[ERROR] Neither 'person_id' nor 'track_id' found in CSV.")

    print(f"[INFO] Source      : {SRC_CSV}")
    print(f"[INFO] Rows        : {rows_before:,}")
    print(f"[INFO] Unique IDs  : {df[id_col].nunique()}")
    print(f"[INFO] ID range    : {df[id_col].min()} - {df[id_col].max()}")
    print(f"[INFO] Copies      : {args.n_copies}")

    combined = duplicate(df, id_col, args.n_copies)

    EXPR_OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] Dataset created")
    print(f"     Rows before : {rows_before:,}")
    print(f"     Rows after  : {len(combined):,}  (= {rows_before:,} x {args.n_copies})")
    print(f"     Unique IDs  : {combined[id_col].nunique()}")
    print(f"     Saved to    : {OUT_CSV}")


if __name__ == "__main__":
    main()
