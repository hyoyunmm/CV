import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

def get_tiny_imagenet(root, split='train', transform=None):
    path = os.path.join(root, 'tiny-imagenet-200')
    
    if split == 'train':
        split_path = os.path.join(path, 'train')
    elif split == 'val':
        # val/annotations.txt를 기준으로 label 재매핑 필요함 (단순 ImageFolder로는 부족할 수도 있음)
        split_path = os.path.join(path, 'val/fixed_val')
    else:
        raise ValueError(f"Unsupported split: {split}")
    
    dataset = ImageFolder(root=split_path, transform=transform, loader=default_loader)
    return dataset
