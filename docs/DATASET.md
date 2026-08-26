# Dataset card

This release distributes **XTile-S2O-Urban76**, the 76-city Sentinel-1/2 full-scene corpus used to train and
evaluate the XTile-S2O translator: co-gridded city-level mosaics, the labelled tile corpus with
pixel-aligned DEM rasters, and the tile- and block-level metadata that define the experimental
protocol.

---

## 1. XTile-S2O-Urban76: the 76-city Sentinel-1/2 full-scene corpus

| | |
|---|---|
| Sensors | Sentinel-1 GRD (IW, VV/VH, **descending**) → Sentinel-2 L2A visible bands |
| Countries / cities | Italy, Poland, Türkiye — 76 cities |
| Tiling | 256 × 256 px at 10 m, stride 160 px → **96 px overlap**; four neighbours form a 2 × 2 block (fused extent 416 px) |
| Tiles | 8,984 labelled pairs; usable after quality filtering: **5,148 train / 1,177 val / 1,234 test** |
| Blocks | 7,553 complete 2 × 2 candidates → **6,179 fully valid (the coherence-training gate; 4,352 in train)** → 6,031 additionally flagged `use_for_block_training` (metadata, not consumed by training) |
| Splits | **city-disjoint**, country-stratified: 37 / 20 / 19 cities (`split` in `patch_index.csv`; repository-side city lists in `manifests/splits/`) |
| Auxiliary raster | Copernicus DEM GLO-30 elevation + slope, per tile, pixel-aligned (slope computed on city mosaics at native resolution) |
| Radiometry | optical clipped to `[0, 4500]` reflectance; SAR clipped VV `[-20, 2]` dB, VH `[-28, -6]` dB — single source of truth in `xtile_s2o/s2o_norm.py` |
| Size | ≈ 13.0 GB imagery + 4.7 GB DEM rasters + 5.1 GB city mosaics |

### Source-product selection

Scenes pair one descending-orbit Sentinel-1 IW GRD VV/VH observation
(`COPERNICUS/S1_GRD`, 10 m) with one Sentinel-2 Level-2A surface-reflectance observation
(`COPERNICUS/S2_SR_HARMONIZED`) per city. The ROI is the main built-up polygon
(`ESA/WorldCover/v100`) within 30 km of the city centroid, buffered by 1,280 m. Pairs were
initially constrained to a **maximum acquisition interval of 7 days**, a scene cloud percentage
≤ 30 %, and an ROI cloud fraction ≤ 0.15 computed per pixel from
`COPERNICUS/S2_CLOUD_PROBABILITY`. For 15 cities no sufficiently cloud-free optical scene existed
inside that window, so the interval constraint was relaxed for those cities in a later selection
round. The delivered distribution is therefore bimodal: **median interval 2.50 days, 52 of 76 pairs within
7 days (23 of them same-day), and a maximum of 65.79 days** concentrated in the relaxed cities
(their median is 20.2 days).

Tiles are cropped locally from the exported city mosaics, not exported independently, so every
tile is traceable to a named source scene, and through `manifests/source_products.csv` to the
ESA product identifier of that scene. The acquisition and preprocessing pipeline itself is not
part of this release; the retained provenance metadata (source scene names, product identifiers,
acquisition times, orbit/polarization metadata, and per-tile pixel origins) is sufficient to reconstruct or extend
the corpus independently under identical acquisition settings.

### Manifest authority and released metadata

The Zenodo `corpus76` archive is the authoritative data distribution. Its `manifests/` directory
contains three canonical files, mirrored byte-for-byte in this repository:

* `patch_index.csv` — 8,984 tile pairs described by 18 fields covering split, city/grid position,
  labels, relative raster paths, pixel origins, and source-mosaic filenames;
* `block_2x2_index.csv` — 7,553 complete 2 × 2 candidates with member ids, labels, and derived
  training/evaluation eligibility flags;
* `source_products.csv` — 76 city-level source pairs described by 11 fields covering scene
  filenames, acquisition times, pair gaps, orbit/polarizations, and Sentinel product identifiers.

The additional files under the GitHub repository's `manifests/` directory are companion assets for
reproducing the paper. They do not extend the contents claimed for the canonical Zenodo archive.

* **Tile annotations** (`manifests/patch_index.csv`, column `coarse_label`) — a six-class quality
  and content scheme: **0 invalid · 1 snow · 2 non-built-up · 3 sparse built-up · 4 dense
  built-up · 5 cloud**. Produced by one annotator and independently verified by a second, with
  disagreements resolved by the verifier; a later automated check reassigned 11 tiles containing
  SAR NaN pixels to class 0. Classes {0, 1, 5} are excluded from all experiments, leaving 7,559
  usable tiles; {3, 4} mark built-up content.
* **Block eligibility metadata** (`manifests/block_2x2_index.csv`) — every geometrically complete
  2 × 2 block with its four member ids and labels, and five **derived** flags: `all_four_valid`
  (6,179), `has_built_up` (6,398), `use_for_block_training` (6,031), `is_tiling_anchor` (2,059)
  and `use_for_tiling_eval` (1,684 = 1,166 train / 254 val / 264 test; the disjoint-cover subset
  on which overlap inconsistency and the seam decomposition are computed — its val and test
  subsets are exactly the paper's frozen 254-block and 264-block disjoint sets; the rule is
  `is_tiling_anchor AND all_four_valid`, without a built-up requirement). These are resolved from the tile annotations and the
  spatial relation — they are eligibility metadata, not a second round of manual annotation.
* **Repository-side full-scene deployment cover** (`manifests/scene_cover_test.csv`) — the exact ordered 412-block
  manifest used for full-scene test inference: 264 disjoint anchors followed by 148 supplemental
  blocks. It contains 1,648 block memberships and covers 1,231 unique usable tiles in 18 of the
  19 assigned test cities; Bytom has no fully usable 2 × 2 block. Construction, overwrite
  semantics, and invariant checks are documented in [`REPRODUCE.md`](REPRODUCE.md).
* **Repository-side per-city tables** — `manifests/city_index.csv` (tile/block counts + source
  pair), `manifests/city_dem_stats.csv` (terrain description), and
  `manifests/splits/{train,val,test}_cities.txt`.

The canonical CSV headers and the component README define the Zenodo tables. The repository-side
[`manifests/schema.yaml`](../manifests/schema.yaml) documents both those canonical fields and the
additional companion manifests used by the released code.

### Re-usability

Because the release contains the **city-level mosaics** as well as the per-tile row/column and
pixel-origin index, the 256-px / stride-160 grid is a choice of this study rather than a property
of the data: any other tile size, overlap or block layout can be regenerated from the same source
scenes. The six-class annotations support content- and quality-conditioned sampling schemes other
than the one adopted here, and the block flags let a different study gate region-level training by
its own criterion.

### Known characteristics that affect evaluation

* **The fixed-band reference OI is a convention artifact, not a seam level.** On strictly
  nominal-stride adjacent pairs (tile displacement exactly 160 px), reference crops are
  bit-identical and the reference overlap inconsistency is exactly 0. The nonzero fixed-band
  aggregate (51.0) arises entirely from pairs involving a scene's clamped final row or column,
  where the nominal 96-px band compares spatially offset pixels (see the paper's Supplementary
  Note S9 and `evaluation/oi_support_sensitivity.py`). Separately, the Salerno and Bytom
  reference mosaics contain genuine multi-date source boundaries; these do not contribute to
  the 264-block OI statistics.
* **Quality labels are coarse and partly heuristic.** They are task-specific quality/content
  filters rather than semantic ground truth.

---

## 2. Provenance, licences and attribution

| Source | Terms | What we redistribute |
|---|---|---|
| **Copernicus Sentinel-1 / Sentinel-2** | Copernicus open data policy — free reuse and redistribution, including of modified data, with attribution | City mosaics and processed tiles (clipped, tiled, normalized). Required notice: *"Contains modified Copernicus Sentinel data (2022–2023)."* |
| **Copernicus DEM GLO-30** (WorldDEM-30) | COP-DEM licence — attribution, liability waiver **and** pass-down clause all required | Derived per-tile elevation and slope rasters, in their own directory with the full COP-DEM licence. Only the **GLO-30** instance is publicly redistributable; COP-DEM-EEA-10 / GLO-10 are not, and none is used here. The mandatory notices are reproduced below. |
| **Our labels, indices, splits** | released with this repository / archived record | patch and block labels, city splits, eligibility manifests |

### Required notices, verbatim

Reproduce these in the record description, in each component's `LICENSE`, and in the paper's data
availability statement.

**Sentinel-derived mosaics and tiles**
> Contains modified Copernicus Sentinel data (2022–2023). Neither the European Commission nor ESA
> incurs any liability for any use of the Copernicus Sentinel data.

**DEM-derived tiles** — all three sentences are mandatory:
> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH
> 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
> The organisations in charge of the Copernicus programme by law or by delegation do not incur any
> liability for any use of the Copernicus WorldDEM-30.
> Any redistribution of these derived tiles is bound by the same obligations.

### Licensing structure — a collective database, not a merged product

The record mixes author-created materials (CC BY 4.0) with Copernicus imagery and DEM
derivatives that keep their own terms; the record-level `DATA_LICENSES.md` and each
component's in-archive `README.txt` state the terms and mandatory notices per component:

```
corpus76/     Sentinel tiles: Copernicus Sentinel source terms + modified-data notice;
              manifests, labels, splits: CC-BY-4.0
dem_tif/      COP-DEM (WorldDEM-30) licence with pass-down obligations
fullscenes/   Copernicus Sentinel source terms + modified-data notice;
              source_products.csv: CC-BY-4.0
```

---

## 3. Intended use and limitations

Translated optical imagery is a *surrogate*: the model may hallucinate plausible structures, and
cross-tile agreement does not certify semantic or physical correctness. The corpus and released code are
intended for methodological research on full-scene translation and cross-tile coherence. Do not
use outputs of this system as sole evidence for operational monitoring, mapping products, or any
decision affecting individuals or property.

---

## 4. Access

```bash
python scripts/download_data.py --target ./data --components corpus76 dem fullscenes
python scripts/verify_data.py  --target ./data
```

Component sizes, URLs and SHA-256 checksums are listed in `scripts/record.json`. The record DOI
is [10.5281/zenodo.22082922](https://doi.org/10.5281/zenodo.22082922); files remain restricted
during peer review and are planned for public release upon manuscript acceptance. See also the
component-level summary in [`DATA_LICENSES.md`](../DATA_LICENSES.md).
