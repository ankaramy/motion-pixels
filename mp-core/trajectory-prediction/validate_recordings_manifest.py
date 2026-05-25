"""
Validate recordings_manifest.csv for completeness and path integrity.

Run from repo root:
    python mp-core/trajectory-prediction/validate_recordings_manifest.py
"""

import sys
from pathlib import Path
import csv
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "mp-data" / "recordings" / "recordings_manifest.csv"

REQUIRED_COLUMNS = {
    "recording_id", "location", "typology", "interior_exterior", "daytime",
    "fps", "video_path", "plan_path", "calibration_path", "tracking_csv_path",
    "encoded_csv_path", "usable", "split", "notes",
}
VALID_SPLITS = {"train", "val", "test", "pending"}
VALID_USABLE = {"yes", "no"}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR  manifest not found: {path}")
        sys.exit(1)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows: list[dict]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    # Column check
    if rows:
        present = set(rows[0].keys())
        missing_cols = REQUIRED_COLUMNS - present
        if missing_cols:
            errors.append(f"Missing columns: {sorted(missing_cols)}")

    # Unique IDs
    ids = [r["recording_id"].strip() for r in rows]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]
    if duplicates:
        errors.append(f"Duplicate recording_id(s): {duplicates}")

    missing_paths: list[str] = []
    for r in rows:
        rid = r["recording_id"].strip()
        usable = r.get("usable", "").strip().lower()
        split = r.get("split", "").strip().lower()

        if usable not in VALID_USABLE:
            errors.append(f"{rid}: usable='{usable}' must be one of {VALID_USABLE}")

        if split not in VALID_SPLITS:
            errors.append(f"{rid}: split='{split}' must be one of {VALID_SPLITS}")

        if usable == "yes":
            enc_path_str = r.get("encoded_csv_path", "").strip()
            if not enc_path_str:
                errors.append(f"{rid}: usable=yes but encoded_csv_path is empty")
            else:
                enc_path = REPO_ROOT / enc_path_str
                if not enc_path.exists():
                    missing_paths.append(f"{rid}: {enc_path_str}")

    if missing_paths:
        for mp in missing_paths:
            errors.append(f"Missing encoded_csv_path -> {mp}")

    return errors, warnings


def print_summary(rows: list[dict], errors: list[str], warnings: list[str]) -> None:
    total = len(rows)
    usable = [r for r in rows if r.get("usable", "").strip().lower() == "yes"]
    pending = [r for r in rows if r.get("split", "").strip().lower() == "pending"]
    split_counts = Counter(r.get("split", "").strip().lower() for r in usable)

    print("=" * 52)
    print("  Recordings Manifest Validation")
    print("=" * 52)
    print(f"  Manifest : {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"  Total recordings   : {total}")
    print(f"  Usable (yes)       : {len(usable)}")
    print(f"  Pending            : {len(pending)}")
    print(f"  Train / Val / Test : {split_counts.get('train',0)} / {split_counts.get('val',0)} / {split_counts.get('test',0)}")
    print()

    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    ! {w}")
        print()

    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    x {e}")
        print()
        print("  RESULT: FAIL")
        print("=" * 52)
        sys.exit(1)
    else:
        print("  RESULT: PASS")
        print("=" * 52)


def main() -> None:
    rows = load_manifest(MANIFEST_PATH)
    errors, warnings = validate(rows)
    print_summary(rows, errors, warnings)


if __name__ == "__main__":
    main()
