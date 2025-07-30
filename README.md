# CV

## 데이터셋
1. CIFAR10 과 CIFAR100 중 선택 해서 진행 (작성되어 있는 코드 활용)
2. TinyImagenet에 대해 직접 데이터셋 코드 작성 후 진행

## 실험 내용
1. image의 attribution을 추출하는 것이 무엇인지 파악 (vanilla gradients)
| 파악 대상                      | 설명                                                   |
|-------------------------------|--------------------------------------------------------|
| 🔍 어떤 위치의 픽셀이 예측에 중요한지 | attribution map은 중요 pixel에 높은 값                    |
| 🔍 모델이 특정 클래스에 주목한 영역  | 예: 고양이 class 예측 시 귀, 눈, 얼굴 주변이 강조됨       |
| 🔍 모델 간 해석 비교              | ResNet18 vs ResNet50 vs ViT 등                         |
| 🔍 데이터셋 간 반응 차이          | CIFAR10 vs Tiny-ImageNet 등                            |


2. timm으로 모델을 바꿔보며 실험

ex)
hyper-parameter 변화에 따른 성능 및 학습(검증) 과정 비교
학습된 모델의 logits이 class 별로 잘 구분 되는지 (PCA, t-SNE 등)
etc

### 구조
```bash
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
```
