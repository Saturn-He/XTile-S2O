# -*- coding: utf-8 -*-
"""Pre-registered terrain-subgroup analysis and DEM gate reporting.

Build symlink subsets for mountainous and flat cities, then run the released
metric implementation with the same conventions as the main evaluation table.
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
E = Path(os.environ.get("XTILE_EVAL_ROOT", REPO_ROOT / "eval")).resolve()
GT = Path(os.environ.get("XTILE_DATA_ROOT", REPO_ROOT / "data" / "corpus76")).resolve() / "test" / "Optical_tif"
SUB = E / "subgroup"
terc_path = Path(os.environ.get("XTILE_TERRAIN_TERCILES",
                                REPO_ROOT / "manifests" / "dem_terrain_terciles.json")).resolve()
terc = json.loads(terc_path.read_text(encoding="utf-8"))
GROUPS = {"mount": set(terc["mountainous"]), "flat": set(terc["flat"])}
city_re = re.compile(r"^(.*?)_r\d+_c\d+")

def first(g):
    r = glob.glob(g); return r[0] if r else None

METHODS = {
    "d0": first(str(E / "e_b8perc_test" / "*_pred_tif")),
    "d1": first(str(E / "d1_dino_test" / "*_pred_tif")),
    "g1": first(str(E / "g1_demcat_test" / "*_pred_tif")),
    "g2": first(str(E / "g2_demca_test" / "*_pred_tif")),
}
print({k: v for k, v in METHODS.items()}, flush=True)
assert all(METHODS.values()), "missing pred dir"

def city_of(fn):
    m = city_re.match(os.path.basename(fn))
    return m.group(1) if m else None

SUB.mkdir(parents=True, exist_ok=True)
# GT subsets
for g, cities in GROUPS.items():
    d = SUB / f"GT_{g}"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in glob.glob(str(GT / "*.tif")):
        if city_of(f) in cities:
            dst = d / os.path.basename(f)
            if not os.path.lexists(dst): os.symlink(f, dst)
            n += 1
    print(f"GT_{g}: {n}", flush=True)
# method subsets
for m, src in METHODS.items():
    for g, cities in GROUPS.items():
        d = SUB / f"{m}_{g}"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in glob.glob(f"{src}/*.tif"):
            if city_of(f) in cities:
                dst = d / os.path.basename(f)
                if not os.path.lexists(dst):
                    os.symlink(f, dst)
                n += 1
        print(f"{m}_{g}: {n}", flush=True)

# Run the released metric implementation per (method, group).
for g in GROUPS:
    pairs = [f"{m}_{g}:{SUB / f'{m}_{g}'}" for m in METHODS]
    cmd = [sys.executable, str(EVAL_DIR / "metrics_s2oit.py"), "--level", "patch",
           "--methods", *pairs, "--optical_dir", str(SUB / f"GT_{g}"),
           "--out_csv", str(SUB / f"metrics_{g}.csv")]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

print("== per-group CSVs ==", flush=True)
for g in GROUPS:
    print((SUB / f"metrics_{g}.csv").read_text(encoding="utf-8"), flush=True)

# Report learned DEM cross-attention gate values.
g2_ckpt = Path(os.environ.get("XTILE_G2_CHECKPOINT",
                              REPO_ROOT / "runs" / "g2_demca" / "checkpoint-last.pth")).resolve()
if g2_ckpt.exists():
    ck = torch.load(g2_ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    gammas = {k: float(v) for k, v in sd.items() if "dem_gamma" in k}
    print("DEM cross-attention gate values (model weights):", gammas, flush=True)
else:
    print(f"DEM gate report skipped: checkpoint not found at {g2_ckpt}", flush=True)
print("SUBGROUP_DONE", flush=True)
