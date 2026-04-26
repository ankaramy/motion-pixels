"""
duplicate_trajectories.py
--------------------------
Overfitting experiment helper.

Duplicates the trajectory data N_COPIES times, giving each copy a unique
ID range so tracks never collide across copies.  Saves the result back to
the same CSV so the rest of the pipeline picks it up automatically.

Usage
-----
  python duplicate_trajectories.py
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE        = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"

CSV_PATH   = OUTPUTS_DIR / "trajectories_world.csv"
BACKUP_PATH = OUTPUTS_DIR / "trajectories_world_ORIGINAL_BACKUP.csv"

N_COPIES = 10


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        sys.exit(f"[ERROR] Not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    rows_before = len(df)

    # Detect which ID column is present
    if "track_id" in df.columns:
        id_col = "track_id"
    elif "person_id" in df.columns:
        id_col = "person_id"
    else:
        sys.exit("[ERROR] Neither 'track_id' nor 'person_id' found in CSV.")

    print(f"[INFO] Loaded {rows_before:,} rows  |  ID column: '{id_col}'")
    print(f"[INFO] Unique IDs in original: {df[id_col].nunique()}")
    print(f"[INFO] ID range: {df[id_col].min()} – {df[id_col].max()}")

    max_id   = int(df[id_col].max())
    stride   = max_id + 1   # offset per copy so IDs never collide

    copies = []
    for i in range(N_COPIES):
        copy = df.copy()
        copy[id_col] = copy[id_col] + i * stride
        copies.append(copy)

    combined = pd.concat(copies, ignore_index=True)
    rows_after = len(combined)

    combined.to_csv(CSV_PATH, index=False)

    print(f"\n[OK] Duplicated {N_COPIES}x")
    print(f"     Rows before : {rows_before:,}")
    print(f"     Rows after  : {rows_after:,}  (= {rows_before:,} × {N_COPIES})")
    print(f"     Unique IDs  : {combined[id_col].nunique()}")
    print(f"     ID range    : {combined[id_col].min()} – {combined[id_col].max()}")
    print(f"\n     Saved to    : {CSV_PATH}")
    print(f"     Original backup: {BACKUP_PATH}")


if __name__ == "__main__":
    main()
