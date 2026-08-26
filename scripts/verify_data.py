# -*- coding: utf-8 -*-
"""Verify an already-downloaded data tree against the manifests in this repository.

Checks (no network access):
  * tile counts per split against manifests/patch_index.csv;
  * every referenced SAR/optical file exists;
  * block index integrity: all four members of each usable block are present;
  * optional local checkpoints load and expose their parameter count.

    python scripts/verify_data.py --target ./data --weights ./weights
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, os.pardir, "manifests")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="./data")
    ap.add_argument("--weights", default="./weights")
    ap.add_argument("--skip-files", action="store_true", help="count rows only, do not stat files")
    args = ap.parse_args()
    fail = 0
    data_root = os.path.join(args.target, "corpus76")
    if not os.path.isdir(data_root):
        data_root = args.target

    rows = list(csv.DictReader(open(os.path.join(MAN, "patch_index.csv"), encoding="utf-8-sig")))
    per_split = {}
    missing = []
    for r in rows:
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
        if args.skip_files:
            continue
        for col in ("sar_path", "optical_path"):
            p = os.path.join(data_root, r[col])
            if r[col] and not os.path.exists(p):
                missing.append(p)
    print(f"patch manifest: {len(rows)} rows  " +
          "  ".join(f"{k}={v}" for k, v in sorted(per_split.items())))
    if missing:
        fail += 1
        print(f"  MISSING {len(missing)} referenced files, e.g. {missing[:3]}")
    elif not args.skip_files:
        print("  all referenced tiles present")

    ok = {r["patch_id"] for r in rows}
    blocks = list(csv.DictReader(open(os.path.join(MAN, "block_2x2_index.csv"),
                                      encoding="utf-8-sig")))
    usable = [b for b in blocks if b["use_for_block_training"] == "true"]
    quad = ["top_left", "top_right", "bottom_left", "bottom_right"]
    broken = [b["block_id"] for b in usable if any(b[f"{q}_patch_id"] not in ok for q in quad)]
    print(f"block index: {len(blocks)} candidates, "
          f"{sum(b['all_four_valid'] == 'true' for b in blocks)} valid, {len(usable)} usable")
    if broken:
        fail += 1
        print(f"  BROKEN {len(broken)} usable blocks reference unknown patches, e.g. {broken[:3]}")
    else:
        print("  every usable block resolves to known patches")

    if os.path.isdir(args.weights):
        try:
            import torch
            for fn in sorted(f for f in os.listdir(args.weights) if f.endswith(".pth")):
                c = torch.load(os.path.join(args.weights, fn), map_location="cpu",
                               weights_only=False)
                if not isinstance(c, dict):
                    raise TypeError(f"{fn}: expected a checkpoint dictionary")
                sd = c.get("model_ema1") or c.get("model") or c.get("state_dict")
                if not isinstance(sd, dict):
                    raise KeyError(f"{fn}: no model_ema1, model, or state_dict mapping")
                tensors = [v for v in sd.values() if hasattr(v, "numel")]
                if not tensors:
                    raise ValueError(f"{fn}: selected state dictionary contains no tensors")
                n = sum(v.numel() for v in tensors) / 1e6
                print(f"weights: {fn}  {n:.1f}M params  ({c.get('description', 'n/a')})")
        except ImportError:
            print("weights: torch not installed, skipped")
    else:
        print(f"weights: {args.weights} not found, skipped")

    print("\nRESULT:", "FAIL" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
