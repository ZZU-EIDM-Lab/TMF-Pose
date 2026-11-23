# Third-party software and copyright notes

The TMF-Pose authors have confirmed that TMF-Pose was **developed by modifying and extending AG-Pose source code**, rather than merely using AG-Pose as an academic reference. The AG-Pose MIT license allows modifications and redistribution provided that its original copyright and permission notice are retained. The exact original MIT text is included at `LICENSES/AG-Pose-LICENSE.txt`. This document tracks other upstream material as well; it is **not** a claim that every embedded third-party file has completed a license audit.

| Component | Evidence in this repository | Upstream / license note |
|---|---|---|
| AG-Pose | TMF-Pose directly modifies and extends AG-Pose source code, including its model and supporting implementation. | [AG-Pose](https://github.com/Leeiieeo/AG-Pose) is under the MIT License; upstream notice preserved in `LICENSES/AG-Pose-LICENSE.txt` (Copyright (c) 2024 Xiao Lin). Retain the full AG-Pose copyright and license notice with the distributed derived implementation; review adapted files and their own headers. |
| PointNet++ / CUDA operations | `model/pointnet2/setup.py` contains a Facebook copyright and MIT-license header; CUDA/C++ source is bundled. | Identify the exact upstream distribution and keep all its applicable notices/license files. The original archive included no standalone PointNet++ license file. |
| Pose evaluation utilities | `utils/evaluation_utils.py`, `utils/housecat6d_eval_utils.py`, and `utils/CATRE_evaluation_utils.py` contain Gorilla Lab, SCUT copyright/MIT headers. | Verify the corresponding upstream repository/license text and preserve required attribution for these files. |
| DINOv2, CLIP, datasets | Referenced as external model/data resources but not distributed here. | Obtain them from their owners and follow the corresponding licenses and data terms. |

**Before making the GitHub repository public:**

1. The paper participants have confirmed agreement to public release. Separately verify any school, laboratory, or collaborator ownership constraints if applicable.
2. Identify every bundled third-party source file and keep the relevant full license/copyright notices.
3. Choose and add a project-wide license for the new TMF-Pose contributions if the rightsholders intend to permit reuse, while retaining upstream AG-Pose and other applicable third-party terms. Do not mistake the AG-Pose MIT text for an automatic license grant by TMF-Pose rightsholders.
4. Review the untouched code and configs for machine-specific paths or sensitive metadata before release.

No global `LICENSE` file was fabricated in this packaging step.

**Attribution note:** Being based on AG-Pose is now disclosed explicitly in `README.md`; the upstream MIT text must remain intact. Attribution in a README alone is not a substitute for the required copyright and license notices.
