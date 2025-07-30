# CV

## 데이터셋
1. CIFAR10 과 CIFAR100 중 선택 해서 진행 (작성되어 있는 코드 활용)
2. TinyImagenet에 대해 직접 데이터셋 코드 작성 후 진행

## 실험 내용
1. image의 attribution을 추출하는 것이 무엇인지 파악 (vanilla gradients)
2. timm으로 모델을 바꿔보며 실험

### 구조
CV_classification/
├── configs/                           ← 모델/데이터셋 별 config yaml
│   ├── cifar10_resnet18.yaml
│   ├── tiny_resnet18.yaml
│   └── ...
│
├── datasets/                          ← 데이터셋 로딩 코드
│
├── models/                            ← ResNet, ViT 등 모델 정의
│
├── train.py                           ← 모델 학습 코드
├── test.py                            ← 모델 평가 코드
│
├── main.py                            ← 학습 전체 파이프라인
│
├── attribution/                       ← attribution 관련 코드
│   ├── vanilla_grad.py
│   ├── run_attribution.py
│   ├── visualize.py
│   └── results/
│       ├── cifar10/
│       │   ├── resnet18/
│       │   │   ├── airplane_1.png
│       │   └── vit_small/
│       └── tiny-imagenet/
│           ├── resnet18/
│           └── vit_base/
│
├── saved_model/
│   ├── cifar10/
│   │   ├── resnet18/
│   │   │   └── best_model.pth
│   └── tiny-imagenet/
│       └── resnet18/
│
├── scripts/
│   ├── train_all.py                   ← 모든 config 학습 자동화 스크립트
│   └── attribution_all.py            ← 학습된 모델 attribution 일괄 실행
