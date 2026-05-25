"""
Build master_dataset.csv by concatenating all usable recordings from the manifest.

Run from repo root:
    python mp-core/trajectory-prediction/build_master_dataset.py

Splitting is recording-level: a recording belongs entirely to train, val, or test.
"""

import sys
import csv
from pathlib import Path
from datetime import date
from collections import Counter

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "mp-data" / "recordings" / "recordings_manifest.csv"
OUT_DIR = REPO_ROOT / "mp-data" / "processed" / "master_dataset"


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR  manifest not found: {path}")
        sys.exit(1)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(rows: list[dict]) -> pd.DataFrame:
    usable = [r for r in rows if r.get("usable", "").strip().lower() == "yes"]
    if not usable:
        print("ERROR  no usable recordings in manifest")
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for r in usable:
        rid = r["recording_id"].strip()
        split = r["split"].strip().lower()
        enc_path = REPO_ROOT / r["encoded_csv_path"].strip()

        if not enc_path.exists():
            print(f"SKIP   {rid}: encoded_csv_path not found ({enc_path})")
            continue

        df = pd.read_csv(enc_path)
        if "recording_id" not in df.columns:
            df.insert(0, "recording_id", rid)
        if "split" not in df.columns:
            df["split"] = split

        print(f"  loaded {rid:20s}  {len(df):6d} rows  split={split}")
        frames.append(df)

    if not frames:
        print("ERROR  no recordings could be loaded")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def write_summary(master: pd.DataFrame, included: list[dict], out_dir: Path) -> None:
    today = date.today().isoformat()
    split_counts = Counter(master["split"])
    rows_per_rec = master.groupby("recording_id").size().to_dict()

    lines = [
        f"# Master Dataset Summary",
        f"",
        f"**Generated:** {today}",
        f"",
        f"## Recordings Included",
        f"",
    ]
    for r in included:
        rid = r["recording_id"].strip()
        n = rows_per_rec.get(rid, 0)
        lines.append(f"- `{rid}` — {n} rows (split: {r['split'].strip()})")

    lines += [
        f"",
        f"## Totals",
        f"",
        f"- Total rows: {len(master)}",
        f"- Train rows: {split_counts.get('train', 0)}",
        f"- Val rows:   {split_counts.get('val', 0)}",
        f"- Test rows:  {split_counts.get('test', 0)}",
        f"",
        f"## Columns",
        f"",
        f"```",
    ]
    for col in master.columns:
        lines.append(f"  {col}")
    lines += ["```", ""]

    summary_path = out_dir / "master_dataset_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary  -> {summary_path.relative_to(REPO_ROOT)}")


def main() -> None:
    rows = load_manifest(MANIFEST_PATH)
    usable_rows = [r for r in rows if r.get("usable", "").strip().lower() == "yes"]

    print(f"\nBuilding master dataset from {len(usable_rows)} usable recording(s)...\n")
    master = build(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "master_dataset.csv"
    master.to_csv(out_csv, index=False)

    print(f"\n  Dataset  -> {out_csv.relative_to(REPO_ROOT)}")
    print(f"  Rows     : {len(master)}")
    print(f"  Columns  : {len(master.columns)}")

    write_summary(master, usable_rows, OUT_DIR)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
