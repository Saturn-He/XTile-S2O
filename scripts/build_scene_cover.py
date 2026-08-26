#!/usr/bin/env python3
"""Build the ordered 2x2 block cover used for full-scene test inference.

The construction matches the paper run: first select disjoint anchors whose
top-left row and column are both even, then greedily add all-valid blocks for
still-uncovered usable tiles. The output row order is operational: when a tile
occurs in multiple blocks, ``blockgroup_infer.py --all_blocks`` retains the
prediction from its last covering row.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


QUADRANT_COLUMNS = (
    "top_left_patch_id",
    "top_right_patch_id",
    "bottom_left_patch_id",
    "bottom_right_patch_id",
)
EXCLUDED_LABELS = {"0", "1", "5"}


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--patch-index", type=Path,
                        default=root / "manifests" / "patch_index.csv")
    parser.add_argument("--block-index", type=Path,
                        default=root / "manifests" / "block_2x2_index.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination CSV; use a temporary path when auditing the release cover")
    args = parser.parse_args()

    split = args.split.strip().lower()
    patches, _ = read_csv(args.patch_index)
    blocks, fieldnames = read_csv(args.block_index)
    usable = {
        row["patch_id"].strip()
        for row in patches
        if row.get("split", "").strip().lower() == split
        and row.get("coarse_label", "").strip() not in EXCLUDED_LABELS
    }

    valid_blocks = []
    for row in blocks:
        if row.get("split", "").strip().lower() != split:
            continue
        patch_ids = [row[column].strip() for column in QUADRANT_COLUMNS]
        if all(patch_id in usable for patch_id in patch_ids):
            valid_blocks.append(row)

    tile_blocks = defaultdict(list)
    for index, row in enumerate(valid_blocks):
        for column in QUADRANT_COLUMNS:
            tile_blocks[row[column].strip()].append(index)

    covered = set()
    chosen = []
    for index, row in enumerate(valid_blocks):
        if int(row["top_left_row"]) % 2 == 0 and int(row["top_left_col"]) % 2 == 0:
            chosen.append(index)
            covered.update(row[column].strip() for column in QUADRANT_COLUMNS)

    for patch_id in sorted(usable - covered):
        candidates = tile_blocks.get(patch_id, [])
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda index: sum(
                valid_blocks[index][column].strip() not in covered
                for column in QUADRANT_COLUMNS
            ),
        )
        if best not in chosen:
            chosen.append(best)
        covered.update(valid_blocks[best][column].strip() for column in QUADRANT_COLUMNS)

    selected = [valid_blocks[index] for index in chosen]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    uncovered = sorted(usable - covered)
    cities = {row.get("city", "").strip() for row in selected}
    print(f"usable tiles:           {len(usable)}")
    print(f"valid complete blocks:  {len(valid_blocks)}")
    print(f"selected cover blocks:  {len(selected)}")
    print(f"covered unique tiles:   {len(covered)}")
    print(f"effective cover cities: {len(cities)}")
    print(f"uncovered usable tiles: {', '.join(uncovered)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
