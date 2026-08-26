# Third-party components and licenses

This repository combines original code by the XTile-S2O authors (MIT, see `LICENSE`) with code
derived from, or interoperating with, the projects below. Licenses were checked against the
`LICENSE` file shipped in each upstream repository.

## Code we derive from (included in this repository)

| Component | Upstream | License | How we use it |
|---|---|---|---|
| `xtile_s2o/model_jit.py`, `denoiser.py`, `main_jit.py`, `engine_jit.py`, `util/` | JiT — Just image Transformers (Tianhong Li, Kaiming He) | MIT © 2025 Tianhong Li | Backbone and training loop; we add SAR/DEM conditioning, cross-patch attention, block/overlap losses, and full-scene inference. Upstream license retained at `docs/LICENSE.JiT-upstream`. |
| JiT upstream references | SiT (`willisma/SiT`), Lightning-DiT (`hustvl/LightningDiT`) | MIT | Architectural lineage acknowledged in the file headers, as in the upstream repository. |

## Code we interoperate with (fetched, not vendored)

The paper's comparison methods build on the upstream projects below. This release does not
include our adapter shims; upstream code is obtained from the original repositories. The table
is retained for licence transparency and for anyone reproducing the comparisons.

| Baseline | Upstream | License |
|---|---|---|
| Pix2Pix, CycleGAN | `junyanz/pytorch-CycleGAN-and-pix2pix` | BSD (© 2017 Jun-Yan Zhu, Taesung Park) |
| CUT | `taesungp/contrastive-unpaired-translation` | BSD — **but** the upstream tree also ships SinCUT files (`models/sincut_model.py`, `models/stylegan_networks.py`) under the NVIDIA Source Code License-NC. We do not use SinCUT; if you vendor this fork, delete those files and the corresponding LICENSE section, otherwise the non-commercial clause propagates to the whole derived work |
| ControlNet | `lllyasviel/ControlNet` | Apache-2.0 |
| Conditional diffusion (Bai *et al.*) | `Coordi777/Conditional-Diffusion-for-SAR-to-Optical-Image-Translation` | see upstream |
| C-DiffSET | `KAIST-VICLab/C-DiffSET` | MIT © 2026 KAIST VICLab |
| S-CycleGAN (Wang *et al.*, IEEE Access 2019) | reimplemented on the CycleGAN framework above | BSD (framework) |

## Pretrained weights we do **not** redistribute

| Weights | Source | Terms | Role |
|---|---|---|---|
| DINOv3 ViT-B/16 | Meta AI (`facebook/dinov3-vitb16-pretrain-lvd1689m`, gated) | DINOv3 License — official source and access notes are linked in `LICENSES/LicenseRef-DINOv3.txt` | Frozen teacher, **training only**; inference does not touch it. Users obtain it from the official gated repository and accept Meta's terms directly. The model identifier is recorded, but the exact Hub revision used for the reported run was not logged. |
| Stable Diffusion 2.1 base | Stability AI | CreativeML Open RAIL++-M | Used only by the latent-diffusion baselines (ControlNet, C-DiffSET) |
| ImageNet/AlexNet-VGG LPIPS, Inception (FID) | `richzhang/PerceptualSimilarity`, `mseitzer/pytorch-fid` | BSD / Apache-2.0 | Evaluation metrics only |

These dependencies are obtained from their original distributors. No trained XTile-S2O
checkpoint is currently distributed in this repository or data record. If such checkpoints are
released later, they will contain only parameters trained by the authors on Sentinel data; the
DINOv3 teacher is discarded after training and is never embedded in the student checkpoint or
used at inference.

## Data products

Sentinel-1/2 and Copernicus DEM GLO-30 each carry their own terms
and mandatory attribution. See `docs/DATASET.md` — read it before redistributing anything
obtained through this repository.

## Licence notices

Bundled licence texts and authoritative external licence references are collected under
`LICENSES/`. When you modify a third-party file, keep its original copyright line and add a
change notice at the top, e.g.
`# Modified from lllyasviel/ControlNet @ <sha> by the XTile-S2O authors, 2026: <what changed>`
— this is an explicit requirement of Apache-2.0 §4(b) and of the OpenRAIL licences, and it is
the condition academic forks most often breach.

## Reporting

If you believe a component is misattributed or a license notice is missing, please open an issue;
we will correct it promptly.
