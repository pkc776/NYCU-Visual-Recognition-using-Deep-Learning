# NYCU Visual Recognition using Deep Learning

This repository contains four labs for the NYCU Computer Vision / Visual Recognition using Deep Learning course. Each lab is organized in its own folder and includes the code needed for training, inference, and generating submission files.

## Labs Overview

| Lab | Folder | Task | Main Method |
| --- | --- | --- | --- |
| Lab 1 | `NYCU-Computer-Vision-2026-HW1-main` | Image Classification | Modified ResNet-50 |
| Lab 2 | `NYCU-Computer-Vision-2026-HW2-main` | Digit Detection | Two-Stage Deformable DETR |
| Lab 3 | `NYCU-Computer-Vision-2026-HW3-main` | Instance Segmentation | Cascade Mask R-CNN + ConvNeXt-V2 |
| Lab 4 | `NYCU-Computer-Vision-2026-HW4-main` | Rain/Snow Image Restoration | PromptIR-based Restoration Model |

## Lab 1: Image Classification

Lab 1 focuses on image classification. The goal is to assign each input image to the correct category. The implementation uses a modified ResNet-50 backbone together with data augmentation and a PyTorch training pipeline to improve validation accuracy. The main components include the dataset loader, model definition, training script, and evaluation script, which produces the final prediction CSV file.

## Lab 2: Digit Detection

Lab 2 focuses on digit detection. The task is to locate digits in an image and predict their classes. This lab uses Two-Stage Deformable DETR with a ResNet-50 backbone, which is suitable for handling long digit-strip images with unusual aspect ratios. The code covers data processing, model construction, training, inference, post-processing, and evaluation, then outputs a JSON file in the required detection submission format.｀

## Lab 3: Instance Segmentation

Lab 3 focuses on instance segmentation. The goal is to detect objects and generate a mask for each individual instance. The implementation is based on Cascade Mask R-CNN in Detectron2 with a ConvNeXt-V2-Base backbone. The workflow includes dataset registration, model training, and test-set inference to generate segmentation predictions.

## Lab 4: Rain/Snow Image Restoration

Lab 4 focuses on rain and snow image restoration. The goal is to recover clean images from degraded rainy or snowy inputs. This lab is built on a PromptIR-based all-in-one restoration framework and adds NAFNet-style residual blocks, residual prediction, EMA training, restoration losses, and physically reasonable data augmentation. The code supports training, multi-GPU training, TTA inference, and validation of the final submission `.npz` file.

## Repository Structure

```text
.
├── NYCU-Computer-Vision-2026-HW1-main/  # Image classification
├── NYCU-Computer-Vision-2026-HW2-main/  # Digit detection
├── NYCU-Computer-Vision-2026-HW3-main/  # Instance segmentation
├── NYCU-Computer-Vision-2026-HW4-main/  # Rain/snow image restoration
└── README.md
```

For detailed environment setup, training commands, and inference commands, please refer to the README inside each lab folder.
