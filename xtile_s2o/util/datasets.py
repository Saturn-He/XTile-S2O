import csv
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import tifffile
import torch


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(root):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    files = [p for p in root_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    return sorted(files)


class ImageDirDataset(Dataset):
    def __init__(self, root, transform=None, mode="RGB"):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.files = _list_images(root)
        if not self.files:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert(self.mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


class PairedImageDirDataset(Dataset):
    def __init__(self, sar_root, opt_root, transform=None):
        self.sar_root = sar_root
        self.opt_root = opt_root
        self.transform = transform
        self.sar_files = _list_images(sar_root)
        self.opt_files = _list_images(opt_root)
        if len(self.sar_files) != len(self.opt_files):
            raise ValueError("SAR and OPT datasets must be the same length.")
        for sar_path, opt_path in zip(self.sar_files, self.opt_files):
            if sar_path.name != opt_path.name:
                raise ValueError(f"Mismatched filenames: {sar_path.name} vs {opt_path.name}")

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_path = self.sar_files[idx]
        opt_path = self.opt_files[idx]
        sar_img = Image.open(sar_path).convert("L")
        opt_img = Image.open(opt_path).convert("RGB")
        if self.transform is not None:
            try:
                sar_img, opt_img = self.transform(sar_img, opt_img)
            except TypeError:
                sar_img = self.transform(sar_img)
                opt_img = self.transform(opt_img)
        return sar_img, opt_img


def _extract_patch_id(filename):
    stem = Path(filename).stem
    if stem.endswith("_SAR"):
        return stem[:-4]
    if stem.endswith("_OPT"):
        return stem[:-4]
    raise ValueError(f"Cannot extract patch_id from filename: {filename}")


def _load_label_map(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    label_map = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_fields = {"patch_id", "split", "coarse_label"}
        if not required_fields.issubset(reader.fieldnames or set()):
            raise ValueError(f"CSV missing required fields: {required_fields}")
        for row in reader:
            patch_id = row["patch_id"]
            label_map[patch_id] = {
                "split": row["split"].strip().lower(),
                "coarse_label": int(row["coarse_label"]),
            }
    return label_map


def _is_kept_label(coarse_label):
    return coarse_label not in {0, 1, 5}


class FilteredPairedImageDirDataset(Dataset):
    def __init__(self, sar_root, opt_root, csv_path, split, transform=None,
                 dem_dir=None, dem_norm=None):
        # When DEM conditioning is enabled, __getitem__ loads <pid>_DEM/_SLOPE,
        # normalizes via dem_norm (train p1/p99 json) and returns the SAR CARRIER
        # [VV,VH,elev,slope] (4ch) instead of 2ch — flag-off path is unchanged.
        self.transform = transform
        self.dem_dir = dem_dir
        self.dem_norm = dem_norm
        self.samples = []
        label_map = _load_label_map(csv_path)
        split = split.lower()

        sar_files = _list_images(sar_root)
        opt_files = _list_images(opt_root)
        opt_by_patch = {_extract_patch_id(p.name): p for p in opt_files}

        for sar_path in sar_files:
            patch_id = _extract_patch_id(sar_path.name)
            meta = label_map.get(patch_id)
            if meta is None or meta["split"] != split or not _is_kept_label(meta["coarse_label"]):
                continue
            opt_path = opt_by_patch.get(patch_id)
            if opt_path is None:
                continue
            self.samples.append((sar_path, opt_path))

        if not self.samples:
            raise ValueError(f"No paired samples found for split={split} after CSV filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sar_path, opt_path = self.samples[idx]
        # SAR: 2-band (VV/VH), float, in dB  -> shape (2,H,W)
        sar = tifffile.imread(str(sar_path)).astype(np.float32)
        if sar.ndim == 2:  # safety: single-band fallback
            sar = sar[None, ...]
        elif sar.shape[-1] in (1, 2, 3):  # HWC -> CHW
            sar = np.transpose(sar, (2, 0, 1))
        # OPT: 3-band RGB reflectance (uint16) -> (3,H,W)
        opt = tifffile.imread(str(opt_path)).astype(np.float32)
        if opt.ndim == 2:
            opt = opt[None, ...]
        elif opt.shape[-1] in (1, 3):
            opt = np.transpose(opt, (2, 0, 1))
        if self.dem_dir is not None:
            from dem_prior import load_dem_pair
            pid = _extract_patch_id(sar_path.name)
            dem = self.dem_norm(load_dem_pair(self.dem_dir, pid))   # (2,H,W) in [-1,1]
            if self.transform is not None:
                sar, opt, dem = self.transform(sar, opt, aux=dem)
            return torch.cat([sar, dem], dim=0), opt                # 4ch carrier
        if self.transform is not None:
            sar, opt = self.transform(sar, opt)
        return sar, opt


class FilteredImageDirDataset(Dataset):
    def __init__(self, root, csv_path, split, transform=None, mode="RGB"):
        self.transform = transform
        self.mode = mode
        self.samples = []
        label_map = _load_label_map(csv_path)
        split = split.lower()

        for path in _list_images(root):
            patch_id = _extract_patch_id(path.name)
            meta = label_map.get(patch_id)
            if meta is None or meta["split"] != split or not _is_kept_label(meta["coarse_label"]):
                continue
            self.samples.append(path)

        if not self.samples:
            raise ValueError(f"No samples found for split={split} after CSV filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if self.mode == "SAR_TIF":
            import numpy as np
            arr = tifffile.imread(str(path)).astype(np.float32)   # SAR 2-band dB
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.shape[-1] in (1, 2, 3, 4) and arr.shape[0] not in (1, 2, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
            arr = np.nan_to_num(arr, nan=-28.0, posinf=2.0, neginf=-28.0)
            image = torch.from_numpy(arr)
        else:
            image = Image.open(path).convert(self.mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name
