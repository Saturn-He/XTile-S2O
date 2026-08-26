# Reproducing the paper

The quantitative tables of the paper map to the scripts below. Paths assume the data record has
been downloaded to `./data` (see `docs/DATASET.md`).

Conventions used throughout the paper and this code:

* **Checkpoint selection is done on validation only.** Baselines: minimum validation LPIPS
  (tie-break FID). The coherence-trained final model: minimum validation FID subject to
  validation OI ≤ 70 — a rule fixed before the checkpoint grid was revealed.
* **Matched sampling noise.** `blockgroup_infer.py` seeds the global PyTorch random-number
  stream once per run and consumes it while traversing the block CSV without shuffling. The
  sampled noise is therefore determined by the run seed, exact CSV row order, and block batch
  size. Factorial arms use the same values for all three, so corresponding tiles receive matched
  noise. Exact replay of the archived sampling sequence requires the released ordered cover and
  `--gen_bsz 32`; hardware or library differences may still prevent byte-identical outputs.
* **Test is touched once**, after a configuration is frozen.

---

## Translation quality and coherence (76-city corpus)

| Paper item | Script | Notes |
|---|---|---|
| Table 3 — auxiliary-prior ablation | `xtile_s2o/main_jit.py` (5 runs) → `evaluation/metrics_s2oit.py --level patch` | flags: `--use_dem {off,concat,ca}`, `--use_dino_prior` |
| Terrain-stratified analysis (Sec. 5.3, Suppl. Note S8) | `evaluation/subgroup_analysis.py` | terciles frozen before results (`manifests/dem_terrain_terciles.json`) |
| Table 4 — translation fidelity | `evaluation/metrics_s2oit.py --level patch --methods <name>:<pred_tif>` | common 1,056-tile subset |
| Table 5 — full-scene coherence | `evaluation/coherence_extra_metrics.py`, `evaluation/allmethods_seam_decomp.py` | 264 disjoint blocks; OI is **pre-fusion** |
| Table 6 — generator–recomposition control | `evaluation/scene_seam_extra.py` | hard stitch vs. cosine feather on identical predictions |
| Table 7 — base–retrofit–joint comparison | `evaluation/coherence_extra_metrics.py` on the fine-tuned arms | 14 block-epochs from the converged base |
| Table 8 — two-seed from-scratch factorial | `evaluation/val2x2_analysis.py` | block-paired bootstrap, 10⁴ resamples; decisions on validation |
| Suppl. Note S1 — recomposition rules | `evaluation/test_fusion_arms.py` | six rules on identical *K* = 8 candidates |
| Suppl. Note S9 — overlap-support sensitivity | `evaluation/oi_support_sensitivity.py` | OI_fixed (all 1,056 edge-sharing pairs) vs. OI_nom (918 strictly nominal-stride pairs); reference OI_nom = 0 exactly |

The frozen disjoint evaluation sets are derived from `manifests/block_2x2_index.csv` as
`is_tiling_anchor AND all_four_valid` (the `use_for_tiling_eval` flag): 254 val blocks
(1,016 tiles; recomposition-rule selection) and 264 test blocks (1,056 tiles; OI and the
factorial study). Unlike `use_for_block_training`, this rule has **no** built-up
requirement — filtering additionally on `has_built_up` yields 253 val blocks and does not
reproduce the paper. The full-scene cover used in the paper extends the disjoint anchors with
remaining fully-usable blocks (test split: 412 cover blocks = 264 anchors + 148
supplemental), reaching 1,231 of the 1,234 usable test tiles (99.8%) in 18 cities; the
three uncovered tiles belong to no fully usable 2×2 block. A tile predicted by more than
one cover block retains the prediction of the last covering block in the fixed cover order
(anchors precede supplemental blocks). The repository-side companion filename is
`manifests/scene_cover_test.csv`. It contains 412 rows, 1,648 block memberships, and
1,231 unique tiles. The cover reaches 18 of the 19 assigned test cities because Bytom has no
fully usable 2x2 block.
In particular, using `--all_blocks` with the complete `block_2x2_index.csv` processes all
943 fully valid test blocks and is not equivalent to the paper's full-scene cover. Validate the
released row order and coverage invariants with `python scripts/verify_scene_cover.py`. To audit
the deterministic two-pass construction, rebuild the ordered block IDs from the public indices:

```bash
python scripts/build_scene_cover.py --output /tmp/scene_cover_test.rebuilt.csv
```

The released cover preserves the original execution CSV byte-for-byte; the rebuilt audit file
uses the current public block-index schema but must have the same ordered `block_id` sequence.
The preserved legacy columns include unpopulated execution-time placeholders (values
`unknown`/`TBD`); they are superseded by the materialized flags in `block_2x2_index.csv`
and are retained only for byte-level fidelity to the execution artifact.

Table 9 (inference cost) reports controlled profiling of the released inference entry points
(`bf16`, 50-step Heun, deployment batch of 32 tiles) on an otherwise idle NVIDIA A6000; no
separate script is required.

Frozen disjoint test evaluation with block-context inference (264 blocks / 1,056 tiles):

```bash
python xtile_s2o/blockgroup_infer.py --resume runs/xtile_s2o/checkpoint-40.pth \
  --gen_split test --gen_bsz 32 --output_dir out/test_blocks
```

Full-scene deployment over the released ordered cover uses `--all_blocks` and follows the
last-covering-block rule:

```bash
python xtile_s2o/blockgroup_infer.py --resume runs/xtile_s2o/checkpoint-40.pth \
  --block_csv manifests/scene_cover_test.csv --all_blocks \
  --gen_split test --gen_bsz 32 --seed 3001 --output_dir out/scene_seed3001
```

### Block-context K=8 candidate generation

The K=8 recomposition study does not use tile-wise inference. Its candidates are generated by
eight seeded calls to `blockgroup_infer.py`, which keeps CPA active by forwarding four tiles per
2x2 block, and are then consumed by `evaluation/test_fusion_arms.py`:

```bash
for seed in $(seq 3001 3008); do
  python xtile_s2o/blockgroup_infer.py \
    --resume runs/xtile_s2o/checkpoint-40.pth \
    --block_csv manifests/scene_cover_test.csv --all_blocks \
    --gen_split test --gen_bsz 32 --seed "${seed}" \
    --output_dir "eval/fstest_seed${seed}"
done
python evaluation/test_fusion_arms.py
```

This is the exact execution structure used by the paper: epoch-40 checkpoint, CPA active after
blocks 5 and 9, DEM carrier enabled from the checkpoint configuration, seeds 3001--3008, the
412-row cover, and 32 blocks per batch. Each seed processes 1,648 block memberships and leaves
1,231 flat tile predictions after ordered overwrite. Preserve the seed, cover row order, and
batch size to reproduce the matched-noise stream.

## Qualitative figures

The qualitative figures (single-tile comparison, full-scene comparisons, seam close-ups) are
assembled from model-generated test-tile predictions together with the released city mosaics and
manifests. The archived release does not distribute the prediction files. The figures use the
display conventions stated in the corresponding captions (per-city 2nd–98th-percentile
reference-valid display stretch; SAR shown with the fixed [-20, 2] dB range). The window-selection
criteria for the seam close-ups are given in the caption of Fig. 8.

## Baselines

The comparison methods are reproduced from their official repositories with a shared data
interface: identical splits, tiling, stitching, and normalization (`xtile_s2o/s2o_norm.py`).
The per-method training configurations are documented in the paper (Sec. 5.1) and its
supplementary notes; the adapter shims themselves are not part of this release.

## Expected wall-clock

| Stage | Hardware | Time |
|---|---|---|
| Base translator (400 epochs) | 1 × A6000 | ≈ 40 h |
| Final coherent model (120 block-epochs) | 1 × A6000 | ≈ 36 h |
| Full-scene inference, 19-city test split | 1 × A6000 | ≈ 1.5 h |
