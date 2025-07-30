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
├── main.py                  ← 학습 파이프라인 시작점
├── visualize.py             ← 학습된 모델에 대해 attribution 시각화 실행
├── train.py                 ← 학습/검증 루프 정의
├── run_experiments.py       ← 여러 모델 반복 실험 스크립트

├── attribution/
│   ├── __init__.py
│   ├── utils.py             ← normalize, heatmap 등 유틸
│   ├── vanilla_grad.py      ← Vanilla gradients 핵심 구현
│   └── visualizer.py        ← attribution 실행 및 시각화 도구

├── datasets/
│   ├── __init__.py
│   ├── augmentation.py      ← get_augmentation()
│   └── factory.py           ← create_dataset(), create_dataloader()

├── configs/                 ← 모델/데이터셋별 YAML 설정
│   ├── cifar10_resnet18.yaml
│   └── tiny_beit_base.yaml

├── saved_model/            ← 학습된 모델 저장
└── results/                ← 실험 결과 기록 (CSV 등)

```
