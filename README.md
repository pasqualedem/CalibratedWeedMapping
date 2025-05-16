# Calibrated Weed Mapping

## Overview

## Preparation
Use the following command to install the required packages:
```bash
uv sync
```

### Dataset processing

```bash
wget http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Orthomosaic/RedEdge.zip
unzip RedEdge.zip -d RoWeeder/dataset

python3 RoWeeder/main.py rotate --root RoWeeder/dataset/RedEdge/000 --outdir RoWeeder/dataset/rotated_ortho/000 --angle -46
python3 RoWeeder/main.py rotate --root RoWeeder/dataset/RedEdge/001 --outdir RoWeeder/dataset/rotated_ortho/001 --angle -46
python3 RoWeeder/main.py rotate --root RoWeeder/dataset/RedEdge/002 --outdir RoWeeder/dataset/rotated_ortho/002 --angle -46
python3 RoWeeder/main.py rotate --root RoWeeder/dataset/RedEdge/003 --outdir RoWeeder/dataset/rotated_ortho/003 --angle -46
python3 RoWeeder/main.py rotate --root RoWeeder/dataset/RedEdge/004 --outdir RoWeeder/dataset/rotated_ortho/004 --angle -46

python3 RoWeeder/main.py patchify --root RoWeeder/dataset/rotated_ortho/000 --outdir RoWeeder/dataset/patches/512/000 --patch_size 512
python3 RoWeeder/main.py patchify --root RoWeeder/dataset/rotated_ortho/001 --outdir RoWeeder/dataset/patches/512/001 --patch_size 512
python3 RoWeeder/main.py patchify --root RoWeeder/dataset/rotated_ortho/002 --outdir RoWeeder/dataset/patches/512/002 --patch_size 512
python3 RoWeeder/main.py patchify --root RoWeeder/dataset/rotated_ortho/003 --outdir RoWeeder/dataset/patches/512/003 --patch_size 512
python3 RoWeeder/main.py patchify --root RoWeeder/dataset/rotated_ortho/004 --outdir RoWeeder/dataset/patches/512/004 --patch_size 512
```