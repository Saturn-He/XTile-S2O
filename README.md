# XTile-S2O

**Cross-Tile Coherent Full-Scene SAR-to-Optical Image Translation**

Official code and dataset manifests for the accompanying manuscript currently under review at
the *ISPRS Journal of Photogrammetry and Remote Sensing*.

> Paper: *submitted to the ISPRS Journal of Photogrammetry and Remote Sensing (DOI to follow)* · Archived data release: [10.5281/zenodo.22082922](https://doi.org/10.5281/zenodo.22082922) *(files restricted during peer review)*

---

## What this repository provides

| Component | Location | Notes |
|---|---|---|
| Model, training and inference code | `xtile_s2o/` | JiT-B/8 flow-matching translator + cross-patch attention (CPA) + block/overlap consistency |
| Joint fidelity--coherence evaluation protocol | `evaluation/` | fidelity metrics, pre-fusion **overlap inconsistency (OI)**, decomposed seam gradients, GDS |
| Dataset manifests and labels | `manifests/` | canonical Zenodo manifest copies plus repository-side cover, split, terrain, and QA metadata |
| Data and cover utilities | `scripts/` | fetches and verifies the archive, reconstructs the scene cover, and checks frozen invariants |

The imagery itself is distributed through the archived data record, not through Git;
`scripts/download_data.py` fetches and verifies it. See [docs/DATASET.md](docs/DATASET.md).

**Release scope.** The repository supports model training and the frozen disjoint
block-context evaluation protocol on the released corpus. The exact ordered 412-block cover
used for the paper's full-scene inference is distributed as
`manifests/scene_cover_test.csv`, enabling the reported deployment geometry to be reconstructed.
Trained XTile-S2O checkpoints and baseline adapter shims are not currently distributed. The
satellite-data acquisition and
preprocessing pipeline (Google Earth Engine export, mosaicking, tiling) is not included.
The canonical Zenodo manifests retain Copernicus product identifiers,
acquisition times, grid offsets, and per-tile pixel origins, so each released tile and block
remains traceable to its source products.

**Manifest authority.** The Zenodo `corpus76` archive is the authoritative dataset distribution.
The same canonical `patch_index.csv` (18 columns), `block_2x2_index.csv`, and
`source_products.csv` (11 columns) are mirrored under `manifests/` in this repository. Additional
repository-side files, including `scene_cover_test.csv`, city/split summaries, DEM statistics,
and `schema.yaml`, support code execution and paper reproduction but are not part of the three
canonical manifest files inside the Zenodo corpus archive.

---

## Headline results

Full-scene coherence on the XTile-S2O-Urban76 corpus (Table 5 of the paper; fixed-support OI
(`OI_fixed`) on 264 disjoint
2×2 test blocks, seam statistics over identical hard-stitched scenes):

| Model | OI ↓ | seam ḡ𝒮 ↓ | interior ḡℐ | GDS ↓ |
|---|---|---|---|---|
| Prior-strengthened base | 139.9 | 213.2 | 87.8 | 2.43 |
| + block consistency (retrofit) | 108.4 | 177.7 | 78.5 | 2.26 |
| **XTile-S2O (joint, from scratch)** | **55.3** | **145.2** | 69.3 | **2.10** |

On the common 1,056-tile fidelity subset, XTile-S2O also attains the best PSNR (20.43 dB)
and SSIM (0.399) among the evaluated methods. The fixed-band reference OI level is 51.0,
which arises entirely from clamped edge pairs; on strictly nominal-stride pairs the
reference OI is exactly 0 (see `evaluation/oi_support_sensitivity.py` and the paper's
Supplementary Note S9). See the paper for the complete fidelity and coherence protocol.

---

## Installation

```bash
git clone https://github.com/Saturn-He/XTile-S2O.git
cd XTile-S2O
conda env create -f environment.yaml     # or: pip install -r requirements.txt
conda activate xtile-s2o
```

Tested with Python 3.10, PyTorch 2.5.1 (CUDA 12.1), on NVIDIA A6000 (48 GB).

## Data

The archived **XTile-S2O-Urban76** record contains the following components (split `tar` archives, parts up to 4.2 GB):

| Component | Contents | Size |
|---|---|---|
| `fullscenes` | the 76 co-gridded city-level Sentinel-1/Sentinel-2 mosaics with acquisition tables | 5.1 GB |
| `corpus76` | 76-city Sentinel-1/2 tiles, six-class quality/content labels, indices | 13.0 GB |
| `dem` | per-tile Copernicus GLO-30 elevation and slope | 4.7 GB |

`fullscenes` is what makes the corpus re-tileable: with the mosaics plus the per-tile
pixel-origin index, the 256-px / stride-160 grid can be replaced by any other tiling.

The archives are deposited in the restricted-review Zenodo record
[10.5281/zenodo.22082922](https://doi.org/10.5281/zenodo.22082922). Verify downloaded parts
against [`manifests/SHA256SUMS.txt`](manifests/SHA256SUMS.txt).

```bash
# Zenodo: scripted download + verification
python scripts/download_data.py --target ./data --components corpus76 dem fullscenes

# alternative manual workflow after downloading all parts into ./data
cd data
sha256sum -c ../manifests/SHA256SUMS.txt
cat xtile_s2o_corpus76.tar.part* > xtile_s2o_corpus76.tar
tar xf xtile_s2o_corpus76.tar    # extracts to corpus76/
cd ..

# check the extracted tree against the manifests
python scripts/verify_data.py --target ./data
```

## Quickstart

Run the paper's frozen disjoint test-block protocol with a trained checkpoint (see the
training recipe below). The checkpoint loader accepts either a checkpoint file or a run
directory containing `checkpoint-last.pth`:

```bash
python xtile_s2o/blockgroup_infer.py \
  --resume runs/xtile_s2o/checkpoint-40.pth \
  --gen_split test --gen_bsz 32 --output_dir out/test_blocks
```

Train from scratch (joint CPA + block consistency, the paper's final recipe):

```bash
torchrun --nproc_per_node=1 xtile_s2o/main_jit.py \
  --model JiT-B/8 --img_size 256 --srgb_target --lambda_perc 0.1 \
  --use_dem concat --use_dino_prior --lambda_dino 0.5 --allow_combined_priors \
  --use_cpa --use_blocks --lambda_block 1.0 --lambda_overlap 0.5 \
  --epochs 120 --output_dir runs/xtile_s2o
```

Every coherence and prior component is behind an explicit flag; with all flags off the
training recipe reduces to the plain backbone baseline.

## Reproducing the paper

[docs/REPRODUCE.md](docs/REPRODUCE.md) maps the released scripts and manifests to the
paper's tables and figures and identifies the additional artifacts required for exact regeneration.

## Repository map

```
xtile_s2o/        model, losses, training engine, block-context inference, diagnostic tile-wise utilities (CPA disabled by design)
evaluation/       fidelity + coherence metrics, seam-gradient decomposition, subgroup analysis
manifests/        canonical Zenodo manifests plus repository-side cover, splits, terrain, and QA metadata
scripts/          data download, scene-cover construction, and verification
docs/             dataset card, reproduction guide, model card, third-party notices
```

## Citation

```bibtex
@unpublished{he2026xtiles2o,
  title  = {XTile-S2O: Cross-Tile Coherent Full-Scene SAR-to-Optical Image Translation},
  author = {He, Jingfei and Shi, Hao and Chen, Yuhang and Zeng, Yunzhuo and Shi, Ruiqi
            and Chen, Liang and Gamba, Paolo},
  year   = {2026},
  note   = {Manuscript under review at the ISPRS Journal of Photogrammetry and Remote Sensing}
}
```

## License and attribution

Our code is released under the MIT License (see `LICENSE`). It builds on the JiT reference
implementation (MIT, © 2025 Tianhong Li); `docs/THIRD_PARTY.md` lists every upstream
component and its terms.

**Built with DINOv3.** A frozen DINOv3 teacher supplies the optical-representation alignment
loss during training only; it is neither redistributed here nor used at inference. Its terms
and the official gated-model source are linked in `LICENSES/LicenseRef-DINOv3.txt`. The model
identifier is recorded in `xtile_s2o/main_jit.py`; the exact Hub revision used for the reported
training run was not logged.

Data products carry component-specific terms and required attributions. See
[DATA_LICENSES.md](DATA_LICENSES.md) and [docs/DATASET.md](docs/DATASET.md) before
redistributing anything obtained here.
