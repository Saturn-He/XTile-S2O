# Model card — XTile-S2O

## Overview

| | |
|---|---|
| Task | SAR → optical image translation over full Sentinel-1 scenes |
| Input | Sentinel-1 GRD VV/VH (descending), 10 m, dB-clipped; Copernicus GLO-30 elevation + slope |
| Output | 3-band optical estimate (Sentinel-2 visible reflectance, `[0, 4500]`) |
| Backbone | JiT-B/8 flow-matching transformer, 12 blocks; base: 133.03 M total / 130.66 M inference parameters; XTile-S2O: 137.76 M total / 135.39 M inference parameters |
| CPA increment | 4.731 M parameters (+3.6% over the inference network) |
| Coherence mechanisms | cross-patch attention after blocks 5 and 9, one-based (zero-initialized gates) + block/overlap consistency loss tied to the inference-time recomposition |
| Auxiliary priors (adopted, not claimed) | DEM channel conditioning; frozen DINOv3 feature alignment (training only) |
| Checkpoints used in the paper | a coherent final model and the prior-strengthened base — EMA weights, selected on validation as documented in `docs/REPRODUCE.md` |

## Training data

76 cities in Italy, Poland and Türkiye; descending-orbit Sentinel-1/2 pairs tiled at 256 px with
stride 160 (96 px overlap). Splits are **city-disjoint** (37 train / 20 val / 19 test cities). Tiles with quality labels
{0, 1, 5} (invalid, snow, cloud) are excluded, leaving 5,148 / 1,177 / 1,234 usable
tiles. Coherence training admits every geometrically complete 2×2 block whose four member
tiles are usable (the `all_four_valid` gate): **4,352 training-split blocks**
(`manifests/block_2x2_index.csv`). The stricter `use_for_block_training` flag
(`all_four_valid AND has_built_up`; 6,031 blocks corpus-wide) is provided as metadata for
content-conditioned alternatives and is not consumed by the paper's training.

## Intended use

Research on full-scene SAR-to-optical translation and cross-tile coherence. The released data,
configurations, and selection rules support training and the frozen disjoint block-context
evaluation. The exact ordered 412-block full-scene cover used in the paper is included in this repository as
`manifests/scene_cover_test.csv`; its construction and invariants are documented in
`docs/REPRODUCE.md`.

## Out-of-scope / known failure modes

* **Not a substitute for optical acquisition.** Outputs are a plausible optical *surrogate*; fine
  structures and radiometry are approximations, and confidently-wrong content is possible.
* **Visible bands only.** No NIR/SWIR, so reflectance indices (NDVI, NDBI) cannot be derived.
* **Domain scope.** Trained on descending-orbit Sentinel data over three countries; other
  orbits, sensors, seasons and biomes are untested.
* **Uncertainty is not a correctness certificate.** Monte-Carlo dispersion supports *selective
  review* (rejecting the most uncertain regions lowers retained reconstruction error) but cannot
  certify individual structures — unusual content is exactly where the translator is uncertain.
* **Operational use requires a human in the loop.** Never use a translated scene as sole evidence
  for operational decisions.

## Evaluation summary

Full-scene coherence (264 disjoint test blocks, matched noise): overlap inconsistency 55.3
(prior-strengthened base 139.9). Translation fidelity on the common test split is competitive but
not state-of-the-art on FID, by design: the protocol reads coherence *jointly with* fidelity
because a uniformly wrong translator can win seam statistics alone.

## Reproducibility notes

Principal full-budget configurations were trained with a single seed; reported intervals are
block-level bootstraps that quantify evaluation-sample uncertainty conditional on the fitted
model, not run-to-run training variability. Checkpoint selection rules and the pre-specified
analysis plan are documented in `docs/REPRODUCE.md`.

## Third-party components

The frozen DINOv3 teacher is used at training time only and is **not** redistributed here; it is
downloaded from its original source. See `docs/THIRD_PARTY.md`.
