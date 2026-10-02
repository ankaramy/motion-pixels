"""
Encoder V3 (manual masks) - encode a single recording.

    python encoder_v3_manual\\encode_recording_v3_manual.py --recording placa_catalunya_01

Writes to:
    new_datasets\\Barcelona_v3_manual_encoded\\<recording_id>\\spatial_v3_manual\\
        trajectories_encoded_v3.csv
        encoding_v3_metadata.json
        mask_alignment_overlay.png
        feature_diagnostics.json
        feature_diagnostics.md

Does NOT modify the old encoder, old encoded CSVs, the master dataset, or any model.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_lib


def main():
    ap = argparse.ArgumentParser(description="Encoder V3 manual - single recording")
    ap.add_argument("--recording", required=True, choices=v3_lib.VALIDATED,
                    help="validated recording id")
    args = ap.parse_args()
    d = v3_lib.encode_recording(args.recording)
    print(f"\nDONE {args.recording}: STATUS={d['status']}")
    for w in d["warnings"]:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
