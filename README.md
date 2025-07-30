# Tiny-ImageNet Classification Experiments


## Overview
다양한 딥러닝 모델(CNN & Transformer 기반)을 Tiny-ImageNet 데이터셋에서 학습 및 평가하여 구조적 차이가 성능에 어떤 영향을 미치는지 비교
- From-scratch training (no pre-trained weights)

## Datasets & Models
- Dataset: Tiny-ImageNet (200 classes, 64×64 images)
- Image Size: Resized to 224×224 during preprocessing
- Models

| Model Name             | Type        | Notes                            |
|------------------------|-------------|----------------------------------|
| `resnet18`             | CNN         | Lightweight baseline             |
| `resnet50`             | CNN         | Deeper ResNet for scaling test   |
| `efficientnet_b0`      | CNN         | Compound-scaled modern CNN       |
| `convnext_base`        | CNN         | ConvNet with Transformer-style   |
| `vit_base_patch16_224` | Transformer | Pure Vision Transformer baseline |

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
- Accuracy / Loss: 자동 기록 및 .json 저장
- Feature Representation: logits.pt, labels.pt 기반 분석 (t-SNE 등 가능)
- Confusion Matrix: confusion_matrix.png 저장
- Error Analysis: 추후 misclassified sample 시각화 가능

