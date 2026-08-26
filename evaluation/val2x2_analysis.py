# -*- coding: utf-8 -*-
"""Pre-registered VALIDATION 2x2 factorial analysis for the from-scratch coherence arms.
Decision layer (per protocol: val decides, test reported once after freeze).

Arms: hascr (neither), hbscr (block only), hcscr (CPA only), hdscr (both),
      plus retrofit reference hbv (fine-tuned Hb) and GT.
Per disjoint val block: OI_l1, OI_dssim (same oi_of as coherence_extra_metrics),
interior gradient g_I (anti-smoothing check), GDS.
Contrasts with block-paired bootstrap 95% CI (10k resamples):
  Hc-Ha (CPA alone), Hb-Ha (block alone), Hd-Hb (CPA|block), Hd-Hc (block|CPA),
  interaction (Hd-Hb-Hc+Ha).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
from coherence_extra_metrics import (QUAD, PID_RC, load_refl, oi_of,
                                     build_mosaic, gds_of)

E = str(Path(os.environ.get("XTILE_EVAL_ROOT", REPO_ROOT / "eval")).resolve())
D = str(Path(os.environ.get("XTILE_DATA_ROOT", REPO_ROOT / "data" / "corpus76")).resolve())
ARMS = {"hascr": f"{E}/bl_hascr_val/pred_tif", "hbscr": f"{E}/bl_hbscr_val/pred_tif",
        "hcscr": f"{E}/bl_hcscr_val/pred_tif", "hdscr": f"{E}/bl_hdscr_val/pred_tif",
        "hbv_ft": f"{E}/bl_hbv_val/pred_tif"}
GT_DIR = f"{D}/val/Optical_tif"
BCSV = str(Path(os.environ.get("XTILE_BLOCK_MANIFEST",
                               REPO_ROOT / "manifests" / "block_2x2_index.csv")).resolve())


def index_tifs(d, suffix):
    return {f.replace(suffix, "").replace(".tif", ""): os.path.join(d, f)
            for f in os.listdir(d) if f.endswith(".tif")}


pred = {a: index_tifs(p, "_fakeB.tif") for a, p in ARMS.items()}
gt = index_tifs(GT_DIR, "_OPT.tif")

rows = []
with open(BCSV, newline="", encoding="utf-8-sig") as fh:
    rd = csv.DictReader(fh)
    cols = {q: next(c for c in rd.fieldnames if q in c.lower() and "patch" in c.lower()) for q in QUAD}
    splitcol = next((c for c in rd.fieldnames if "split" in c.lower()), None)
    for r in rd:
        if splitcol and (r.get(splitcol) or "").strip().lower() != "val":
            continue
        pids = {q: (r.get(cols[q]) or "").strip() for q in QUAD}
        tl = pids["top_left"]; m = PID_RC.search(tl + "_")
        if not m or int(m.group(1)) % 2 or int(m.group(2)) % 2:
            continue
        if any(p not in gt for p in pids.values()):
            continue
        if any(any(p not in pred[a] for p in pids.values()) for a in ARMS):
            continue
        rows.append(pids)
print(f"paired disjoint val blocks across ALL arms: {len(rows)}", flush=True)


def interior_grad(mos, cnt):
    g = mos.mean(-1)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    it = cnt == 1
    return float(mag[it].mean())


keys = list(ARMS) + ["gt"]
per = {a: {"oi_l1": [], "oi_ds": [], "gI": [], "gds": []} for a in keys}
for i, pids in enumerate(rows):
    tilesets = {a: {q: load_refl(pred[a][pids[q]]) for q in QUAD} for a in ARMS}
    tilesets["gt"] = {q: load_refl(gt[pids[q]]) for q in QUAD}
    for a in keys:
        l1, ds = oi_of(tilesets[a])
        mos, cnt = build_mosaic(tilesets[a])
        per[a]["oi_l1"].append(l1); per[a]["oi_ds"].append(ds)
        per[a]["gI"].append(interior_grad(mos, cnt)); per[a]["gds"].append(gds_of(mos, cnt))
    if i % 50 == 0:
        print(f"  {i}/{len(rows)}", flush=True)

arr = {a: {k: np.array(v) for k, v in d.items()} for a, d in per.items()}
print("\n===== VAL 2x2 TABLE (means over %d paired blocks) =====" % len(rows))
print(f"{'arm':8s} {'OI_l1':>8s} {'OI_dssim':>9s} {'g_I':>8s} {'GDS':>7s}")
for a in keys:
    print(f"{a:8s} {arr[a]['oi_l1'].mean():8.2f} {arr[a]['oi_ds'].mean():9.4f} "
          f"{arr[a]['gI'].mean():8.2f} {arr[a]['gds'].mean():7.3f}")

rng = np.random.default_rng(77)
n = len(rows)
B = 10000
idx = rng.integers(0, n, size=(B, n))


def boot_ci(diff_vec):
    boots = diff_vec[idx].mean(axis=1)
    return diff_vec.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


CONTRASTS = [
    ("CPA alone       (Hc-Ha)", "hcscr", "hascr", None, None),
    ("block alone     (Hb-Ha)", "hbscr", "hascr", None, None),
    ("CPA | block     (Hd-Hb)", "hdscr", "hbscr", None, None),
    ("block | CPA     (Hd-Hc)", "hdscr", "hcscr", None, None),
    ("interaction (Hd-Hb-Hc+Ha)", "hdscr", "hbscr", "hcscr", "hascr"),
    ("joint vs retrofit (Hd-HbFT)", "hdscr", "hbv_ft", None, None),
]
for metric in ("oi_l1", "oi_ds", "gI"):
    print(f"\n===== contrasts on {metric} (mean [95% CI], negative = first arm lower) =====")
    for name, a, b, c, d in CONTRASTS:
        if c is None:
            vec = arr[a][metric] - arr[b][metric]
        else:
            vec = arr[a][metric] - arr[b][metric] - arr[c][metric] + arr[d][metric]
        m, lo, hi = boot_ci(vec)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {name:28s} {m:+9.3f} [{lo:+9.3f}, {hi:+9.3f}]  {sig}")

out = f"{E}/val2x2_oi.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "n_blocks", "OI_l1", "OI_dssim", "g_I", "GDS"])
    for a in keys:
        w.writerow([a, n, arr[a]["oi_l1"].mean(), arr[a]["oi_ds"].mean(),
                    arr[a]["gI"].mean(), arr[a]["gds"].mean()])
np.savez(f"{E}/val2x2_perblock.npz", **{f"{a}_{k}": arr[a][k] for a in keys for k in arr[a]})
print(f"\nsaved {out} + val2x2_perblock.npz")
print("VAL2X2_DONE", flush=True)
