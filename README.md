# Calibrated Weed Mapping

## Overview

## Preparation
Use the following command to install the required packages, through [uv](https://docs.astral.sh/uv/)
```bash
uv sync
```

## Dataset
This repo relies on the preprocessing done in [RoWeeder](https://github.com/yourusername/RoWeeder).

You can either directly download the alredy preprocessed version through our script (recommended):
```bash
bash download.sh
```

Or you can clone and preprocess the dataset using RoWeeder. First clone it, then:

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

## Experiments

You can reproduces all the experiments using the commands in `scripts.sh`, they are divided in three phases:

### Training

Example with MobileNetv4 and Focal Loss

```bash
python main.py train --model mobilenetv4 --loss focal --gamma 2.0
```

The training will produce a model checkpoint at `weights/` depending on the parameters passed.

### Calibration

You can reproduce the calibration step using the following command, after training:

```bash
python main.py calibrate --model mobilenetv4 --calibration_tecnique temperature_scaling --num_epochs 30 --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
```

### Test

Run the following command to test the original model:

```bash
python main.py test --model mobilenetv4 --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
```

Or the calibrated one:

```bash
python main.py evaluate --model mobilenetv4 --calibration_tecnique temperature_scaling --calibration_params weights/mobilenetv4_calibrated_n30_temperature_scaling_ckpt_mobilenetv4_focal_gamma2.pkl --checkpoint weights/mobilenetv4_focal_gamma2.0.pth
```