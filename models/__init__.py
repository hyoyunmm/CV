from .resnet import *
import timm

# 모델 불러오기
def get_model(name, num_classes):
    name = name.lower()

    # 커스텀 구현된 모델
    if name == 'resnet18':
        return ResNet18(num_classes)
    elif name == 'resnet34':
        return ResNet34(num_classes)
    elif name == 'resnet50':
        return ResNet50(num_classes)

    # timm 모델 사용
    if name in timm.list_models(pretrained=True):
        model = timm.create_model(name, pretrained=True)
        # 분류 헤드 재설정
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=num_classes)
        elif hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            in_features = model.head.in_features
            model.head = torch.nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError(f"{name} 모델의 classifier head 수정 방식이 정의되어 있지 않습니다.")
        return model

    raise ValueError(f"Unknown model name: {name}")