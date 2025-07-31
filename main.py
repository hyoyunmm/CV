# docker start -ai dsba_cv_pretraining_container
# cd /workspace/DSBA_Pretraining/CV_classification
# 학습 따로 : python main.py --config configs/tiny_imagenet_res18.yaml
# 시각화 따로 : python visualize.py --config configs/tiny_imagenet_res18.yaml

# python main.py --config configs/tiny_imagenet_vit_tiny_patch16_224.yaml
# python main.py --config configs/tiny_imagenet_resnet18.yaml
import argparse
import yaml
from datasets.augmentation import get_augmentation

import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging

from train import fit, log_final_result ##
from models import *
from datasets import create_dataset, create_dataloader
from log import setup_default_logging

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

## model selecting
from models import resnet
# from models import vit?
def build_model(model_name, num_classes):
    model_dict = {
        'resnet18' : resnet.ResNet18,
        'resnet34' : resnet.ResNet34,
        'resnet50' : resnet.ResNet50,
        'resnet101' : resnet.ResNet101,
        'resnet152' : resnet.ResNet152
        # etc
    }
    if model_name in model_dict:
        return model_dict[model_name](num_classes=num_classes)
    else:
        # timm을 통한 모델 생성
        try:
            model = timm.create_model(model_name, pretrained=True)
            # classification head 수정
            if hasattr(model, 'reset_classifier'):
                model.reset_classifier(num_classes=num_classes)
            elif hasattr(model, 'head'):
                in_features = model.head.in_features
                model.head = torch.nn.Linear(in_features, num_classes)
            else:
                raise NotImplementedError(f"{model_name}의 classifier head 수정 방법을 지정해주세요.")

            return model

        except Exception as e:
            raise ValueError(f"Invalid model name: {model_name}. Not found in custom models or timm.\n{e}")
        
    #if model_name not in model_dict:
    #    raise ValueError(f"Invalid model name : {model_name}")
    #else:
    #    return model_dict[model_name](num_classes=num_classes)

def run(args):
    # make save directory
    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    model = build_model(args.model_name, num_classes=args.num_classes) ## dynamic model selecting
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    ## debug2 : 모델 구조 확인
    print("[DEBUG] Model structure:")
    print(model)

    # load dataset
    ## trainset, testset = create_dataset(datadir=args.datadir, dataname=args.dataname, aug_name=args.aug_name)
    # load dataset
    train_transform = get_augmentation(args.aug_name, is_train=True, image_size=args.image_size)
    test_transform  = get_augmentation('test', is_train=False, image_size=args.image_size)
    trainset = create_dataset(datadir=args.datadir, dataname=args.dataname, split='train', transform=train_transform)
    testset  = create_dataset(datadir=args.datadir, dataname=args.dataname, split='val', transform=test_transform)


    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=32, shuffle=False)
    
    ## debug5: 클래스 수 및 레이블 범위 확인
    print(f"[DEBUG] num_classes: {args.num_classes}")
    print(f"[DEBUG] Total train samples: {len(trainloader.dataset)}")
    targets = [target for _, target in trainloader.dataset]
    print(f"[DEBUG] Label range: {min(targets)} to {max(targets)}")

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr)

    ## debug4: lr 확인
    for param_group in optimizer.param_groups:
        print(f"[DEBUG] Learning rate: {param_group['lr']}")


    # scheduler
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # initialize wandb
    wandb.init(name=args.exp_name, project='DSBA-study', config=args)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = args.epochs, 
        savedir      = savedir,
        log_interval = args.log_interval,
        device       = device,
        exp_name = args.exp_name) ## fit -> test 에서 디버깅 위함
    
    ## Attribution 시각화 -> 학습과 분리시킬 것
    #from attribution.visualizer import AttributionVisualizer

    #savedir = os.path.join('attribution','saved_attribution', args.exp_name)
    #visualizer = AttributionVisualizer(
    #    model=model,
    #    model_name=args.model_name,  # config.yaml에 지정된 모델 이름 필요!
    #    dataloader=testloader,
    #    device=device
    #)
    #visualizer.visualize_and_save(
    #    save_dir=savedir)

if __name__=='__main__':
    ## config 인자만 먼저 추출
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Classification for Computer Vision")
    
    parser.add_argument('--config', type=str, help='Path to config.yaml file')

    # exp setting
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--datadir',type=str,default='/data',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')

    # datasets
    parser.add_argument('--dataname',type=str,default='CIFAR100',choices=['CIFAR10','CIFAR100'],help='target dataname')
    parser.add_argument('--num-classes',type=int,default=100,help='target classes')

    ## model name
    parser.add_argument('--model-name', type=str, default='ResNet18', help='name of model(upper OK)')

    ## image size
    parser.add_argument('--image-size', type=int, default=224, help='input image size (e.g. 32 for CIFAR, 224 for ViT)')


    # optimizer
    parser.add_argument('--opt-name',type=str,choices=['SGD','Adam'],help='optimizer name')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')

    # scheduler
    parser.add_argument('--use_scheduler',action='store_true',help='use sheduler')

    # augmentation
    parser.add_argument('--aug-name',type=str,choices=['default','weak','strong'],help='augmentation type')

    # train
    parser.add_argument('--epochs',type=int,default=50,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')

    # seed
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')
    
    ## config selecting
    if config_args.config:
        with open(config_args.config, 'r') as f:
            config = yaml.safe_load(f)
        parser.set_defaults(**config)
        
    args = parser.parse_args()

    print(args)
    run(args)
