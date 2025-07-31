import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import pandas as pd

def log_final_result(exp_name, acc, loss, path='results/results.csv'): ## 저장경로
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([{
        'exp_name': exp_name,
        'accuracy': acc,
        'loss': loss
    }])
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def train(model, dataloader, criterion, optimizer, log_interval: int, device: str) -> dict:   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):

        ## debug1 : 입력 데이터 형태와 레이블 확인용
        if idx == 0:
            print(f"[DEBUG] inputs shape: {inputs.shape}")
            print(f"[DEBUG] targets shape: {targets.shape}")
            print(f"[DEBUG] targets[:10]: {targets[:10]}")

        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets) 

        ## debug3: loss 값 확인
        # print(f"[DEBUG] Loss value at step {idx}: {loss.item():.4f}")
  
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
        
        batch_time_m.update(time.time() - end)
    
        if idx % log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        acc        = acc_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = inputs.size(0) / batch_time_m.val,
                        rate_avg   = inputs.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
   
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
from collections import OrderedDict
import numpy as np
import os

def test(model, dataloader, criterion, log_interval: int, device: str, savedir: str = './results_analysis', exp_name: str = 'model') -> dict:
    correct = 0
    total = 0
    total_loss = 0

    # 분석용 저장 리스트
    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # predict
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            # 기록용
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # acc
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)

            if idx % log_interval == 0 and idx != 0:
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' %
                             (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))

    acc = correct / total
    loss = total_loss / len(dataloader)

    # 🔽 결과 저장
    os.makedirs(savedir, exist_ok=True)
    np.savez(
        os.path.join(savedir, f'{exp_name}_result.npz'),
        preds=np.array(all_preds),
        probs=np.array(all_probs),
        labels=np.array(all_labels)
    )

    return OrderedDict([('acc', acc), ('loss', loss)])


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, 
    epochs: int, savedir: str, log_interval: int, device: str, exp_name: str
) -> None:

    best_acc = 0
    step = 0
    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device)
        torch.cuda.empty_cache()  # test 직전에 추가
        eval_metrics = test(model, testloader, criterion, log_interval, device,
                            savedir='results_analysis', exp_name=exp_name)

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']
        # 마지막 epoch 도달 시 항상 저장
        os.makedirs(savedir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(savedir, f'last_model.pt'))

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))

    # 실험 결과 기록
    log_final_result(
        exp_name=savedir.split('/')[-1],  # 'saved_model/모델이름'에서 이름만 추출
        acc=eval_metrics['acc'],
        loss=eval_metrics['loss']
    )
    
    # 마지막 epoch 도달 시 항상 저장
    torch.save(model.state_dict(), os.path.join(savedir, f'last_model.pt'))

    # -------- logits 저장 (t-SNE용) -------- #
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    feature_dir = os.path.join(savedir, 'features')
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(logits, os.path.join(feature_dir, 'logits.pt'))
    torch.save(labels, os.path.join(feature_dir, 'labels.pt'))
    _logger.info(f"Saved logits & labels to: {feature_dir}")

