# Tiny-ImageNet Classification Experiments


## Overview
다양한 딥러닝 모델(CNN & Transformer 기반, 특히 **경량모델**)을 Tiny-ImageNet 데이터셋에서 학습 및 평가하여 구조적 차이가 성능에 어떤 영향을 미치는지 비교
- From-scratch training (no pre-trained weights)

## Datasets & Models
- Dataset: Tiny-ImageNet (200 classes, 64×64 images)
- Image Size: Resized to 224×224 during preprocessing
- Models

| Model Name                | Type        | Notes                            |
|---------------------------|-------------|----------------------------------|
| `efficientnet_b0`         | CNN         | Compound-scaled modern CNN       |
| `mobilenetv2_100`         | CNN         | Lightweight mobile-friendly CNN  |
| `resnet18`                | CNN         | Lightweight baseline             |
| `vit_tiny_patch16_224`    | Transformer | Small Vision Transformer variant |


- Training configuration

| Item          | Value                                     |
| ------------- | ----------------------------------------- |
| Epochs        | 30                                        |
| Batch Size    | 32                                        |
| Optimizer     | Adam                                      |
| LR            | 0.001                                      |
| Scheduler     | X (optional config)                       |
| Loss Function | CrossEntropyLoss                          |
| Augmentation  | Resize + Normalize (RandAugment optional) |


## Codebase Structure
```bash
CV_classification/
├── configs/                # *.yaml 설정
├── datasets/               # augmentation, factory
├── datasets_imagenet/      # tiny_imagenet.py
├── models/                 # resnet.py (및 timm 사용)
├── main.py
├── train.py
└── requirements.txt

```
- parallel execution
```bash
python main.py --config configs/tiny_imagenet_resnet50.yaml
```
```bash
python main.py --config configs/tiny_imagenet_efficientnet_b0.yaml
```

## Evaluation & Visualization
```bash
results/
└── tiny_imagenet_resnet50/
    ├── best_model.pt
    ├── best_results.json
    ├── log.txt
    ├── features/
    │   ├── logits.pt
    │   └── labels.pt
    └── ... (optional: confusion_matrix.png, npz files)
```

| Model Name             | Turning Point | Val Loss | Val Acc | Val Acc* | Val Acc@*Epoch | Test Loss | Test Acc |
|------------------------|----------------|----------|---------|----------|------------------|-----------|----------|
| `efficientnet_b0`      | 6              | 0.578    | 0.6614  | 0.6614   | 6                | 2.250     | 0.6555   |
| `mobilenetv2_100`      | 8              | 0.632    | 0.6261  | 0.6261   | 8                | 2.228     | 0.6164   |
| `resnet18`             | 5              | 0.542    | 0.6723  | 0.6723   | 5                | 2.180     | 0.6689   |
| `vit_tiny_patch16_224` | 12             | 0.693    | 0.6025  | 0.6025   | 12               | 2.380     | 0.5897   |

| Model Name             | Params (M) | Model Size (MB) | Val Acc    | Test Acc   | 특징 요약                                    |
| ---------------------- | ---------- | --------------- | ---------- | ---------- | ---------------------------------------- |
| `efficientnet_b0`      | 5.3        | \~20            | 66.14%     | 65.55%     | 균형잡힌 구조 (depth/width/resolution scaling) |
| `mobilenetv2_100`      | 3.5        | \~14            | 62.61%     | 61.64%     | 매우 경량, 모바일 최적화, 성능은 다소 낮음                |
| `resnet18`             | 11.7       | \~44            | **67.23%** | **66.89%** | 가장 높은 성능, 고전적 구조지만 여전히 강력                |
| `vit_tiny_patch16_224` | 5.7        | \~21            | 60.25%     | 58.97%     | Transformer 기반, 작은 데이터에 비효율적             |



