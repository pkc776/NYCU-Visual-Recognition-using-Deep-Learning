# Introduction
This repository contains the implementation of the image classification pipeline for the NYCU Computer Vision HW1. The backbone relies on a `ModifiedResNet50` to balance parameter efficiency with performance alongside robust augmentations.

# Environment Setup
1. Use Conda environments to create local container mapping. Minimum requirements point at Python 3.9+.
```bash
conda create -n hw1 python=3.9 -y
conda activate hw1
pip install torch torchvision Pillow pandas tqdm
```
2. Put the `dataset` at the root folder so that it creates `./data`.
3. Run the train script.

# Usage
## Training
To execute the neural classification training procedure on GPU directly, do:
```bash
python src/train.py --epochs 30 --batch_size 128 --lr 0.001
```
This builds into `checkpoints/best_model.pth`.

## Inference
To calculate test datasets predictions and generate `.csv`, evaluate with:
```bash
python src/eval.py
```
This builds `prediction.csv` correctly.

# Performance Snapshot
During initial epoch testing, our `ModifiedResNet50` backbone generated validation accuracy at **85.7%** effectively surpassing weak-baseline utilizing conservative parameter changes and strong PyTorch optimizations.
This parameter scaling enables quick optimization using `torch.amp` features.
