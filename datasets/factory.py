import os
from torch.utils.data import DataLoader

#
from torchvision import datasets
from torchvision import transforms
from datasets.augmentation import default_augmentation, test_augmentation, weak_augmentation, strong_augmentation
from datasets_imagenet.tiny_imagenet import get_tiny_imagenet  # ✅ 새로 import

def create_dataset(datadir, dataname, split, transform):
    if dataname == 'CIFAR10':
        is_train = (split == 'train')
        return datasets.CIFAR10(root=datadir, train=is_train, transform=transform, download=True)

    elif dataname == 'CIFAR100':
        is_train = (split == 'train')
        return datasets.CIFAR100(root=datadir, train=is_train, transform=transform, download=True)

    elif dataname == 'tiny_imagenet':  
        return get_tiny_imagenet(root=datadir, split=split, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataname}")


#def create_dataset(datadir: str, dataname: str, aug_name: str = 'default'):
#    trainset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
#        root      = os.path.join(datadir,dataname), 
#        train     = True, 
#        download  = True, 
#        transform = __import__('datasets').__dict__[f'{aug_name}_augmentation']()
#    )
#    testset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
#        root      = os.path.join(datadir,dataname), 
#        train     = False, 
#        download  = True, 
#        transform = __import__('datasets').__dict__['test_augmentation']()
#    )
#    return trainset, testset

def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 2
    )
