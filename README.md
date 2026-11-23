# TMF-Pose: A Text-Guided Multi-Modal Fusion Method for Category-Level Object Pose Estimation

Official PyTorch implementation of **[A Text-guided Multi-modal Fusion Method for Category-level Object Pose Estimation](https://doi.org/10.3724/SP.J.1089.2026-00041)** (*Journal of Computer-Aided Design & Computer Graphics*, 2026).

TMF-Pose introduces text-guided fusion of RGB, point-cloud, and semantic features for category-level 6D object pose estimation. This implementation is developed by modifying and extending [AG-Pose](https://github.com/Leeiieeo/AG-Pose).

## Environment Settings

Use the Python, PyTorch, and CUDA versions from the original working experiment environment. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

This implementation also requires the PointNet++ CUDA extension and two pretrained backbones:

- **PointNet++:** CUDA/C++ source is provided in `model/pointnet2/`. Its compiled extension `pointnet2._ext` must be available. The archived `setup.py` references `_ext_src/`, whereas the distributed source files are in `src/`; take this into account if rebuilding the extension in a new environment.
- **DINOv2 (ViT-S/14):** obtain the [DINOv2 source](https://github.com/facebookresearch/dinov2) and update the local `torch.hub.load(...)` path in `model/Net.py` for your environment.
- **CLIP (ViT-B/16):** place the [pretrained model files](https://huggingface.co/openai/clip-vit-base-patch16) under `model/clip-vit-base-patch16/` as expected by `model/Net.py`.

## Data Processing

### NOCS (CAMERA25 and REAL275)

Download and preprocess NOCS following [DPDN](https://github.com/JiehongLin/Self-DPDN). Prepare segmentation predictions as described in the [AG-Pose README](https://github.com/Leeiieeo/AG-Pose#data-processing).

Organize the data under `PROJECT_DIR/data/`:

```text
data/
├── camera/
│   ├── train/                    # CAMERA training images and labels
│   ├── val/                      # CAMERA validation images
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── val_list_all.txt
├── real/
│   ├── train/                    # REAL training images and labels
│   ├── test/                     # REAL test images
│   ├── train_list.txt
│   ├── train_list_all.txt
│   └── test_list_all.txt
├── camera_full_depths/          # Preprocessed CAMERA *_composed.png depths
├── obj_models/
│   ├── camera_train.pkl
│   ├── real_train.pkl
│   └── ...
├── segmentation_results/
│   ├── CAMERA25/
│   │   └── results_*.pkl
│   └── REAL275/
│       └── results_*.pkl
└── gts/
```

For each training image, the NOCS loader expects the corresponding `*_color.png`, `*_mask.png`, and `*_label.pkl` files, together with depth images. The CAMERA loader uses the preprocessed depth files in `camera_full_depths/`. Set both `train_dataset.dataset_dir` and `test_dataset.dataset_dir` in `config/REAL/camera_real.yaml` to your `data/` directory. The evaluation split is selected by `test_dataset.dataset_name` (`real` or `camera`).

### HouseCat6D

Download [HouseCat6D](https://sites.google.com/view/housecat6d) and organize it as follows (for example under `PROJECT_DIR/data/HouseCat6D/`):

```text
HouseCat6D/
├── scene*/                     # Training scenes
│   ├── rgb/                    # <frame>.png
│   ├── depth/                  # <frame>.png
│   ├── instance/               # <frame>.png
│   ├── labels/                 # <frame>_label.pkl
│   ├── intrinsics.txt
│   └── meta.txt
├── val_scene*/                 # Validation scenes, if provided
├── test_scene*/                # Test scenes; same per-scene layout
└── obj_models_small_size_final/
```

The loader reads `scene*/rgb/` for training and `test_scene*/rgb/` for evaluation. It matches RGB frames to `depth/`, `instance/`, and `labels/` by filename, and loads camera intrinsics from each scene's `intrinsics.txt`. Set `train_dataset.dataset_dir` and `test_dataset.dataset_dir` in `config/HouseCat6D/housecat6d.yaml` to the **HouseCat6D root**, not to an individual scene.

## Train

**NOCS (CAMERA + REAL):**

```bash
python train.py --config config/REAL/camera_real.yaml
```

**HouseCat6D:**

```bash
python train_housecat6d.py --config config/HouseCat6D/housecat6d.yaml
```

## Evaluate

**NOCS (REAL275 by default):**

```bash
python test.py --config config/REAL/camera_real.yaml --test_epoch 30
```

**HouseCat6D:**

```bash
python test_housecat6d.py --config config/HouseCat6D/housecat6d.yaml --test_epoch 150
```

The original evaluation scripts load `epoch_<N>.pt` from these locations (the epoch numbers above are examples; checkpoints are not included):

```text
log/camera_real/当前最优/epoch_<N>.pt   # NOCS test.py
log/housecat6d/ckpt/epoch_<N>.pt     # HouseCat6D test_housecat6d.py
```

## Results

The paper reports the following results at the **5° / 2 cm** threshold:

| Dataset | REAL275 | CAMERA25 | HouseCat6D |
|:--|--:|--:|--:|
| TMF-Pose | **61.4%** | **78.8%** | **22.6%** |

## Citation

```bibtex
@article{zhao2026tmfpose,
  title={A Text-guided Multi-modal Fusion Method for Category-level Object Pose Estimation},
  author={Zhao, Hongru and Wang, Kaidi and Fang, Guotao and Hu, Shizhe and Chen, Bingxi and Xu, Mingliang},
  journal={Journal of Computer-Aided Design & Computer Graphics},
  year={2026},
  doi={10.3724/SP.J.1089.2026-00041}
}
```

## Acknowledgements

TMF-Pose is developed by modifying and extending the open-source implementation of [AG-Pose](https://github.com/Leeiieeo/AG-Pose). We thank its authors and the developers of the other open-source components used in this project. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attributions.

## License

This project is released under the [MIT License](LICENSE). Original copyright notices and applicable third-party license texts are preserved in [LICENSES/](LICENSES/).
