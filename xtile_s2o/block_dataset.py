"""
block_dataset.py -- 2x2 block-paired dataset for the cross-patch coherence
innovations (cross-patch self-attention + block-mosaic / overlap losses).

Base-agnostic: import `BlockPairedDataset`, `block_collate` from any training
entry point. It reuses the same tif loading +
coarse_label gating as `FilteredPairedImageDirDataset`, but delivers the FOUR
adjacent tiles of a 2x2 block together (fixed quadrant order TL,TR,BL,BR) so the
block ops see a spatially coherent block.

Tiling geometry (from dataset construction): tile 256, stride 160, overlap 96,
2x2 block canvas = 416x416. Constant top-left (x,y) offset of each tile in the
canvas:  TL=(0,0)  TR=(160,0)  BL=(0,160)  BR=(160,160).

IMPORTANT (coherence): the four tiles must stay spatially aligned, so this
dataset does NOT apply independent random spatial augmentation per tile. Pass a
DETERMINISTIC normalize-only transform (the eval transform), or None (raw->tensor
with the same dB/reflectance handling as the patch dataset). Spatial-augmentation
data coverage is provided by the per-patch minibatch of the mixed-batch schedule;
block-level shared augmentation can be added later as an ablation.
"""
import csv
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")
# constant top-left (x, y) offset of each tile inside the 416x416 block canvas
BLOCK_OFFSETS = ((0, 0), (160, 0), (0, 160), (160, 160))
KEEP_EXCLUDE = {0, 1, 5}  # coarse_label values dropped (identical to patch baselines)


def _extract_patch_id(filename):
    stem = Path(filename).stem
    if stem.endswith("_SAR") or stem.endswith("_OPT"):
        return stem[:-4]
    return stem


def _index_by_patch_id(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    idx = {}
    for p in root.iterdir():
        if p.suffix.lower() in IMG_EXTENSIONS:
            idx[_extract_patch_id(p.name)] = p
    return idx


def _load_label_map(csv_path):
    label_map = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patch_id")
            if pid is None:
                continue
            try:
                cl = int(row["coarse_label"])
            except (KeyError, ValueError, TypeError):
                cl = -999
            label_map[pid] = {
                "split": (row.get("split") or "").strip().lower(),
                "coarse_label": cl,
            }
    return label_map


def _detect_columns(fieldnames):
    """Map the 4 quadrant patch-id columns (+ split / block_id cols) from a block
    CSV header. Uses FULL quadrant phrases (top_left ... bottom_right) so the
    2-letter forms can't collide with unrelated columns (e.g. 'bl' in 'block_id').
    Real header: top_left_patch_id, ..., bottom_right_patch_id."""
    raw = list(fieldnames or [])
    low = {c.lower().strip(): c for c in raw}
    cols = {}
    for q in QUADRANTS:  # 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        cand = None
        for key, orig in low.items():           # strict: '<quadrant>..._patch_id'
            if q in key and "patch" in key:
                cand = orig
                break
        if cand is None:                          # fallback: quadrant + id (not block_id)
            for key, orig in low.items():
                if q in key and "id" in key and key != "block_id":
                    cand = orig
                    break
        if cand is None:
            raise ValueError(
                f"BlockPairedDataset: cannot find patch-id column for quadrant '{q}'. "
                f"Header was: {raw}"
            )
        cols[q] = cand
    split_col = None
    for key, orig in low.items():
        if "split" in key:
            split_col = orig
            break
    cols["__split__"] = split_col
    cols["__block_id__"] = low.get("block_id")
    return cols


class BlockPairedDataset(Dataset):
    """Yields the four tiles of a valid 2x2 block.

    __getitem__ -> dict(
        sar      = FloatTensor (4, 2, 256, 256)  # TL,TR,BL,BR
        opt      = FloatTensor (4, 3, 256, 256)
        offsets  = LongTensor  (4, 2)            # (x,y) top-left in 416 canvas
        block_id = str)

    A block is kept iff all four patch_ids are present on disk, are in the
    requested split, and have coarse_label not in {0,1,5} (matching the patch
    baselines). Block-row `split` column is used when present; otherwise the
    per-patch label split is used.
    """

    def __init__(self, sar_root, opt_root, block_csv, label_csv, split,
                 transform=None, require_all_valid=True,
                 dem_dir=None, dem_norm=None):
        # When DEM conditioning is enabled, each tile is emitted as the
        # 4-ch conditioning CARRIER [VV,VH,elev,slope] (same convention as the patch
        # dataset); requires a transform that accepts aux= (PairedTrainTransform).
        self.transform = transform
        self.dem_dir = dem_dir
        self.dem_norm = dem_norm
        if dem_dir is not None:
            assert transform is not None, "DEM carrier needs an aux-capable transform"
        self.split = split.lower()
        self.samples = []  # (block_id, [4 sar Paths], [4 opt Paths])

        sar_idx = _index_by_patch_id(sar_root)
        opt_idx = _index_by_patch_id(opt_root)
        label_map = _load_label_map(label_csv)

        n_rows = n_split = n_complete = 0
        with Path(block_csv).open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = _detect_columns(reader.fieldnames)
            for row in reader:
                n_rows += 1
                if cols["__split__"] is not None:
                    if (row.get(cols["__split__"]) or "").strip().lower() != self.split:
                        continue
                n_split += 1
                pids = [(row.get(cols[q]) or "").strip() for q in QUADRANTS]
                if any(not pid for pid in pids):
                    continue
                ok = True
                sar_paths, opt_paths = [], []
                for pid in pids:
                    sp, op, meta = sar_idx.get(pid), opt_idx.get(pid), label_map.get(pid)
                    if sp is None or op is None or meta is None:
                        ok = False
                        break
                    if cols["__split__"] is None and meta["split"] != self.split:
                        ok = False
                        break
                    if require_all_valid and meta["coarse_label"] in KEEP_EXCLUDE:
                        ok = False
                        break
                    sar_paths.append(sp)
                    opt_paths.append(op)
                if not ok:
                    continue
                n_complete += 1
                bid = (row.get(cols["__block_id__"]) if cols["__block_id__"] else None) \
                    or "__".join(pids)
                self.samples.append((bid, sar_paths, opt_paths))

        self.stats = {"rows": n_rows, "in_split": n_split, "valid_blocks": n_complete}
        if not self.samples:
            raise ValueError(
                f"BlockPairedDataset: no valid 2x2 blocks for split={self.split} "
                f"(rows={n_rows}, in_split={n_split}). Check block_csv columns / paths."
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_sar(path):
        a = tifffile.imread(str(path)).astype(np.float32)  # SAR 2-band (VV,VH) dB
        if a.ndim == 2:
            a = a[None, ...]
        elif a.shape[-1] in (1, 2, 3) and a.shape[0] not in (1, 2, 3):
            a = np.transpose(a, (2, 0, 1))
        return np.nan_to_num(a, nan=-28.0, posinf=2.0, neginf=-28.0)

    @staticmethod
    def _load_opt(path):
        a = tifffile.imread(str(path)).astype(np.float32)  # OPT 3-band RGB reflectance
        if a.ndim == 2:
            a = a[None, ...]
        elif a.shape[-1] in (1, 3) and a.shape[0] not in (1, 3):
            a = np.transpose(a, (2, 0, 1))
        return a

    def __getitem__(self, idx):
        bid, sar_paths, opt_paths = self.samples[idx]
        sars, opts = [], []
        for sp, op in zip(sar_paths, opt_paths):
            sar = self._load_sar(sp)
            opt = self._load_opt(op)
            if self.dem_dir is not None:
                from dem_prior import load_dem_pair
                pid = _extract_patch_id(sp.name)
                dem = self.dem_norm(load_dem_pair(self.dem_dir, pid))  # (2,H,W) [-1,1]
                sar, opt, dem = self.transform(sar, opt, aux=dem)
                sar = torch.cat([sar, dem], dim=0)                     # 4-ch carrier
            elif self.transform is not None:
                sar, opt = self.transform(sar, opt)
            else:  # raw -> tensor (no normalization); prefer passing the eval transform
                sar = torch.from_numpy(sar)
                opt = torch.from_numpy(opt)
            sars.append(sar)
            opts.append(opt)
        return {
            "sar": torch.stack(sars, 0),       # (4,2,256,256)
            "opt": torch.stack(opts, 0),       # (4,3,256,256)
            "offsets": torch.tensor(BLOCK_OFFSETS, dtype=torch.long),  # (4,2)
            "block_id": bid,
        }


def block_collate(batch):
    """Collate G blocks -> sar (G,4,2,256,256), opt (G,4,3,256,256).
    At the call site, flatten to (4G,...) for the backbone via
    `sar.view(-1, *sar.shape[2:])`; keep G to regroup for the block ops."""
    return {
        "sar": torch.stack([b["sar"] for b in batch], 0),
        "opt": torch.stack([b["opt"] for b in batch], 0),
        "offsets": torch.stack([b["offsets"] for b in batch], 0),
        "block_id": [b["block_id"] for b in batch],
    }
