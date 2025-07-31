from torchvision import transforms
from RandAugment import RandAugment

def get_mean_std(dataname: str): ## 데이터셋별 각 채널 평균 (R, G, B), 표준편차
    dataname = dataname.lower()
    if dataname == 'cifar10':
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataname == 'cifar100':
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Default: ImageNet
    
def get_augmentation(aug_name: str, is_train: bool = True, image_size: int = 224, dataname: str = "imagenet"):
    mean, std = get_mean_std(dataname)

    if not is_train:
        return test_augmentation(image_size, mean, std)

    if aug_name == 'default':
        return default_augmentation(image_size, mean, std)
    elif aug_name == 'weak':
        return weak_augmentation(image_size, mean, std)
    elif aug_name == 'strong':
        return strong_augmentation(image_size, mean, std)
    else:
        raise ValueError(f"Unknown augmentation: {aug_name}")

## 데이터셋 권장 mean, std로 적용

def weak_augmentation(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Pad(4),
        transforms.RandomCrop(image_size, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def strong_augmentation(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Pad(4),
        transforms.RandomCrop(image_size, fill=128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform.transforms.insert(0, RandAugment(n=3, m=9))
    return transform



def default_augmentation(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def test_augmentation(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform
