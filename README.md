# TMF-Pose: Text-Guided Multi-Modal Fusion for Category-Level Object Pose Estimation

Official source code for the paper **A Text-guided Multi-modal Fusion Method for Category-level Object Pose Estimation**.

**Paper:** [Journal of Computer-Aided Design & Computer Graphics](https://www.jcad.cn/en/article/doi/10.3724/SP.J.1089.2026-00041)  
**DOI:** [10.3724/SP.J.1089.2026-00041](https://doi.org/10.3724/SP.J.1089.2026-00041)

## Overview

TMF-Pose estimates category-level 6D object poses using RGB images, point-cloud geometry, and text-based semantic priors. The method combines text-guided multi-modal feature fusion, instance-adaptive keypoint detection, geometric-aware feature aggregation, and shape reconstruction.

The TMF-Pose implementation was developed by **modifying and extending the publicly released source code of [AG-Pose](https://github.com/Leeiieeo/AG-Pose)** (CVPR 2024). TMF-Pose incorporates the research contributions described in the accompanying paper. We acknowledge AG-Pose as the foundation of this implementation and preserve its original MIT license notice in [`LICENSES/AG-Pose-LICENSE.txt`](LICENSES/AG-Pose-LICENSE.txt). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

This repository preserves the original TMF-Pose experiment source files. Algorithm implementations, training and evaluation scripts, CUDA source, and YAML configurations have **not been modified** for this release. Only generated files were removed and repository documentation was added.

## Project structure

```text
TMF-Pose/
├── config/
│   ├── REAL/camera_real.yaml
│   └── HouseCat6D/housecat6d.yaml
├── model/
│   ├── Net.py
│   ├── Net_modules.py
│   ├── modules.py
│   ├── inference.py
│   ├── pointnet2/            # Python modules and original CUDA/C++ source
│   └── vnn_net/
├── provider/                 # Dataset loaders
├── utils/                    # Evaluation and training utilities
├── train.py                  # CAMERA/REAL training
├── test.py                   # NOCS evaluation
├── train_housecat6d.py      # HouseCat6D training
├── test_housecat6d.py       # HouseCat6D evaluation
├── requirements.txt
├── .gitignore
├── THIRD_PARTY_NOTICES.md
└── LICENSES/
    └── AG-Pose-LICENSE.txt
```

## Environment

The original archive does not contain an environment lockfile or validated package-version table. Use the Python, PyTorch, CUDA, and compiler versions from your working experiment environment when available. The commands below document dependencies; they are **not** a claim of verification in a clean environment.

Install a CUDA-compatible [PyTorch](https://pytorch.org/get-started/locally/) and torchvision build that matches your driver and toolkit. Then install the source-inferred dependencies:

```bash
python -m pip install -r requirements.txt
```

### External backbone files

The source expects additional local resources that are **not included** in this repository:

- **DINOv2 ViT-S/14:** `model/Net.py` loads `dinov2_vits14` via `torch.hub.load(..., source='local')` from an author-specific absolute checkout path. See [DINOv2](https://github.com/facebookresearch/dinov2). Users on another machine must point this call at their own local DINOv2 checkout.
- **CLIP ViT-B/16:** `model/Net.py` loads the processor and weights from `./model/clip-vit-base-patch16`. Obtain the corresponding [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) files and place them at the expected directory, or adapt the two `from_pretrained` paths in a local checkout.
- **PointNet++ CUDA extension:** The model imports `pointnet2._ext`. The original source and original `setup.py` are retained under `model/pointnet2/`. The original `setup.py` refers to `_ext_src/`, while the provided archive has `src/`; the release deliberately does **not** change this file. Use the compiled extension from a working environment or adjust the build configuration in your own checkout when a rebuild is needed.

These files, the pretrained checkpoints, and the datasets are not bundled in the archive. Installation and rebuild details are environment-dependent.

## Datasets

The code includes loaders for **NOCS CAMERA25 / REAL275** and **HouseCat6D**. Download and prepare the datasets following the providers' terms and their official documentation. This repository does not redistribute the datasets.

In `config/REAL/camera_real.yaml`, the `train_dataset.dataset_dir` and `test_dataset.dataset_dir` fields must refer to the NOCS dataset root. The loader refers to paths such as:

```text
<DATA_ROOT>/camera/train_list.txt
<DATA_ROOT>/real/train_list.txt
<DATA_ROOT>/obj_models/camera_train.pkl
<DATA_ROOT>/obj_models/real_train.pkl
<DATA_ROOT>/segmentation_results/REAL275/results_*.pkl
<DATA_ROOT>/segmentation_results/CAMERA25/results_*.pkl
```

The data loaders also require the associated RGB images, depths, masks, annotations, and model information.

In `config/HouseCat6D/housecat6d.yaml`, point the dataset roots to your HouseCat6D directory. The loader searches `scene*/rgb/` and `test_scene*/rgb/` along with the related annotation and calibration files.

**Reproduction note:** The original YAML files include absolute paths used on the authors' experimental machines. They are intentionally preserved. Update paths in your local copy when transferring the code to a different machine.

## Training

Run from the repository root after preparing the data, backbones, and PointNet++ operator.

**NOCS CAMERA + REAL:**

```bash
python train.py --config config/REAL/camera_real.yaml --gpus 0
```

**HouseCat6D:**

```bash
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --gpus 0
```

Both training scripts set the checkpoint directory to `log/<config-stem>/ckpt/` and save epoch checkpoints. Batch sizes, maximum epochs, and scheduler settings are defined by the YAML configurations. Preserve those settings when reproducing the recorded experiments.

## Evaluation

**REAL275 (the supplied NOCS config selects `dataset_name: real`):**

```bash
python test.py --config config/REAL/camera_real.yaml --test_epoch 30 --gpus 0
```

The `--test_epoch 30` value is an example argument, not a guarantee that this checkpoint is supplied. **The unmodified `test.py` expects checkpoints in `log/camera_real/当前最优/epoch_<N>.pt`, whereas `train.py` saves to `log/camera_real/ckpt/`.** Place the intended checkpoint in the directory expected by the original evaluation script, or adjust that path in your local checkout.

**HouseCat6D:**

```bash
PYTHONPATH="$PWD:$PWD/model:$PWD/provider:$PWD/utils" \
  python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150 --gpus 0
```

The `PYTHONPATH` environment setting above makes the unmodified imports visible. `test_housecat6d.py` expects `log/housecat6d/ckpt/epoch_<N>.pt`. The epoch shown is an example; pretrained checkpoints are not included.

The standalone `model/inference.py` is preserved from the original archive, but it is not documented here as a separately verified inference pipeline.

## Published results

The paper reports the following **5°/2 cm** results:

| Dataset | TMF-Pose |
|:--|--:|
| REAL275 | 61.4% |
| CAMERA25 | 78.8% |
| HouseCat6D | 22.6% |

These values are from the published article, not from a new reproduction conducted for this repository package.

## Citation

If you use this code, please cite:

```bibtex
@article{zhao2026tmfpose,
  title={A Text-guided Multi-modal Fusion Method for Category-level Object Pose Estimation},
  author={Zhao, Hongru and Wang, Kaidi and Fang, Guotao and Hu, Shizhe and Chen, Bingxi and Xu, Mingliang},
  journal={Journal of Computer-Aided Design & Computer Graphics},
  year={2026},
  doi={10.3724/SP.J.1089.2026-00041}
}
```

Volume, issue, and page numbers are intentionally omitted until confirmed by the publisher.

## Acknowledgements and licenses

This repository is a **modified and extended implementation of [AG-Pose](https://github.com/Leeiieeo/AG-Pose)**. We thank the AG-Pose authors for releasing their work and retain the original AG-Pose copyright notice and MIT License in [`LICENSES/AG-Pose-LICENSE.txt`](LICENSES/AG-Pose-LICENSE.txt). TMF-Pose contributes additional research methods described in its paper. The README and code license are separate from the paper citation requirement. Some embedded PointNet++ and evaluation utility files also include third-party copyright or MIT-license headers.

Please review [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before public redistribution. **The AG-Pose MIT license is not presented as a license for all original TMF-Pose contributions.** A project-wide license has not been selected in this package; this requires approval from the relevant copyright holders.

## Reproducibility note

This package is a **source-preserving GitHub distribution**, not a reconstructed or revalidated environment. It removes cached bytecode, training logs, and a machine-specific compiled `.egg`, but retains every original Python, YAML, C++, CUDA, and header file unchanged. See the separately provided file-integrity report for the SHA-256 comparison against the source archive.
