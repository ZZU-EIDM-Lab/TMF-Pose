# TMF-Pose: Text-Guided Multi-Modal Fusion for Category-Level Object Pose Estimation

Official source code for **TMF-Pose**, accompanying the paper:

> **A Text-Guided Multi-Modal Fusion Method for Category-Level Object Pose Estimation**  
> *Journal of Computer-Aided Design & Computer Graphics*  
> [Paper / DOI: 10.3724/SP.J.1089.2026-00041](https://www.jcad.cn/article/doi/10.3724/SP.J.1089.2026-00041)

TMF-Pose combines RGB image, point-cloud, and textual semantic information for category-level 6D object pose estimation. The implementation includes text-guided feature fusion, instance-adaptive keypoint detection, geometric feature aggregation, and pose/size prediction. The provided training and evaluation scripts target the NOCS CAMERA/REAL data and HouseCat6D.

## Repository structure

```text
TMF-Pose/
├── config/
│   ├── REAL/camera_real.yaml         # CAMERA + REAL training; REAL evaluation
│   └── HouseCat6D/housecat6d.yaml    # HouseCat6D configuration
├── model/
│   ├── Net.py                        # TMF-Pose network and text encoder
│   ├── Net_modules.py                # Feature fusion, keypoints, aggregation
│   ├── modules.py                    # Visual/point-cloud components
│   ├── pointnet2/                   # PointNet++ modules and CUDA source
│   └── vnn_net/                     # Vector-neuron point-cloud modules
├── provider/                        # Dataset loaders
├── utils/                           # Training, metrics and utilities
├── train.py                          # CAMERA/REAL training
├── test.py                           # CAMERA/REAL evaluation
├── train_housecat6d.py              # HouseCat6D training
├── test_housecat6d.py               # HouseCat6D evaluation
├── requirements.txt
└── README.md
```

## 1. Environment setup

A Linux machine with an NVIDIA GPU, an appropriate CUDA toolkit, and a compatible C++ compiler is needed for the bundled PointNet++ CUDA operations. Python **3.10 is a suggested starting point**, not a verified requirement; the archive contains a prebuilt PointNet++ artifact labelled for Python 3.10, but it does not record the original complete environment.

```bash
conda create -n tmf-pose python=3.10 -y
conda activate tmf-pose
```

**First** install matching CUDA-enabled PyTorch and torchvision packages using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Then install the Python packages imported by this project:

```bash
python -m pip install -r requirements.txt
```

The dependencies in `requirements.txt` were identified from imports in the supplied source. **Their versions have not been validated together.** In particular, PyTorch, `torch-geometric`, CUDA and the compiled PointNet++ operator must be compatible. Consult the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if additional compiled extensions are needed.

### Required models and CUDA operator

The following resources are **not included** in the supplied source archive:

1. **DINOv2 (ViT-S/14).** In `model/Net.py`, the DINOv2 backbone is loaded with `torch.hub.load(..., 'dinov2_vits14', source='local')` from an author-specific absolute directory. Obtain the appropriate [DINOv2 source and weights](https://github.com/facebookresearch/dinov2), then update that path in `model/Net.py` to point to your own local checkout. Both provided YAML configurations select `rgb_backbone: dino`.
2. **CLIP ViT-B/16.** `model/Net.py` calls `CLIPProcessor.from_pretrained()` and `CLIPModel.from_pretrained()` on `./model/clip-vit-base-patch16`. Place the matching locally saved [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) processor and model files at that location (or update both calls in the source).
3. **PointNet++ CUDA extension.** The implementation imports `pointnet2._ext`. The archive contains CUDA/C++ source under `model/pointnet2/src/` and an environment-specific `.egg`, but no portable installed extension. **The supplied `model/pointnet2/setup.py` references an absent `_ext_src/` directory instead of the included `src/` directory**, so its build configuration must be corrected before attempting to compile/install the extension against your PyTorch/CUDA environment. Do not rely on the bundled prebuilt `.egg` for portability.

**Note:** `requirements.txt` does not download pretrained weights, compile this custom operator, or supply datasets.

## 2. Dataset preparation

Download the datasets from their respective providers, following their access and usage conditions. Dataset files and segmentation predictions are not included in this repository.

### NOCS: CAMERA25 / REAL275

The source expects one NOCS data root containing (at a minimum, for the corresponding tasks):

```text
<DATA_ROOT>/
├── camera/
│   └── train_list.txt
├── real/
│   └── train_list.txt
├── obj_models/
│   ├── camera_train.pkl
│   └── real_train.pkl
└── segmentation_results/
    ├── CAMERA25/results_*.pkl
    └── REAL275/results_*.pkl
```

The loaders also read the images, masks, depth maps and annotation files referenced by these lists/prediction files. Edit `train_dataset.dataset_dir` and `test_dataset.dataset_dir` in `config/REAL/camera_real.yaml` to point to your actual root.

The supplied `config/REAL/camera_real.yaml` uses `camera_real` for training (combined synthetic CAMERA and labeled REAL samples), and `real` for testing. To evaluate CAMERA25 instead, the `test_dataset.dataset_name` and corresponding files must be configured appropriately.

### HouseCat6D

The loader looks for training scenes containing `scene*/rgb/` and test scenes containing `test_scene*/rgb/`, together with associated per-scene calibration and annotations. Set `train_dataset.dataset_dir` and `test_dataset.dataset_dir` in `config/HouseCat6D/housecat6d.yaml` to your dataset root.

## 3. Training

Run commands from the **repository root**, after configuring the datasets and the pretrained models.

**CAMERA + REAL:**

```bash
python train.py --gpus 0 --config config/REAL/camera_real.yaml
```

**HouseCat6D:**

```bash
python train_housecat6d.py --gpus 0 --config config/HouseCat6D/housecat6d.yaml
```

The scripts save model checkpoints as `log/<config-name>/ckpt/epoch_<N>.pt`. Their learning rates, batch sizes, seeds and training epochs are controlled by the YAML configuration files.

## 4. Evaluation

Check that the requested checkpoint exists before running evaluation.

**REAL275 (as configured in the provided NOCS YAML):**

```bash
python test.py --gpus 0 --config config/REAL/camera_real.yaml --test_epoch 30
```

**Important:** The supplied `test.py` loads checkpoints from `log/camera_real/当前最优/epoch_<N>.pt`, **not** the training output directory `log/camera_real/ckpt/`. Place the selected checkpoint in its expected folder or adjust `cfg.ckpt_dir` in `test.py`. The example epoch number is only a command example, not a claim that this checkpoint is provided.

**HouseCat6D (entry-point command):**

```bash
python test_housecat6d.py --gpus 0 --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```

**Known source issue:** As supplied, `test_housecat6d.py` imports `Net` without adding the `model/` directory to `sys.path`. This import path must be checked and corrected before this entry point can execute. The supplied script expects its checkpoint at `log/housecat6d/ckpt/epoch_<N>.pt`. The example epoch number matches the script's default; the weights are not included in the archive.

## 5. Reproduction notes

- The original archive does **not** contain a tested environment lockfile or reproducible package version list. The provided `requirements.txt` is an import-based starting point.
- The source includes hard-coded local paths, missing external model files and a PointNet++ CUDA build-path discrepancy; these must be resolved in the checkout before training/evaluation can run.
- The training, evaluation and inference scripts have **not** been executed as part of preparing this README. Successfully installing the Python packages alone does not validate full reproduction.
- Checkpoint weights and datasets are not distributed with this README. Avoid committing large model binaries, dataset copies or experimental logs unless intentionally hosting them separately.

## Acknowledgements and upstream code

This repository includes and adapts category-level pose estimation components associated with [AG-Pose (CVPR 2024)](https://github.com/Leeiieeo/AG-Pose), including instance-adaptive keypoint detection (IAKD), geometric-aware feature aggregation (GAFA), and related training/data-processing infrastructure. We thank the AG-Pose authors for making their implementation available. TMF-Pose additionally incorporates text-guided multi-modal fusion and other modifications described in our paper.

The repository also contains components carrying upstream copyright headers (including PointNet++ CUDA extension sources and Gorilla Lab evaluation utilities). **Preserve those notices and comply with all applicable upstream license terms.** Before releasing the repository publicly, review the provenance of copied/adapted files and include the required upstream license texts and attributions. AG-Pose is distributed under the MIT License; its license and copyright notices should not be discarded when reproducing its MIT-licensed code.

## Results and pretrained checkpoints

Results are reported in the [TMF-Pose paper](https://doi.org/10.3724/SP.J.1089.2026-00041). This source archive does not include model checkpoints or training logs, and no verified public checkpoint download URL has been supplied. Only add numerical tables and download links here after checking them against the published paper and hosted files.

## Citation

If this project helps your research, please cite the associated paper:

[A Text-Guided Multi-Modal Fusion Method for Category-Level Object Pose Estimation](https://doi.org/10.3724/SP.J.1089.2026-00041), *Journal of Computer-Aided Design & Computer Graphics*, DOI: `10.3724/SP.J.1089.2026-00041`.

## License

The source archive does not include a project-wide license file, so **no license for the TMF-Pose project as a whole is asserted here**. This does not remove the license obligations of upstream code: preserve notices and include any applicable license texts for third-party or adapted sources (including MIT-licensed AG-Pose code, where reused). Obtain authorization from the relevant copyright holders before selecting a new license for original TMF-Pose code.
