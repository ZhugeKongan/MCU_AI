#引入必须的包
import torch
import torchvision
import torchvision.transforms as transforms
from models.mobilenet_v2 import MobileNetV2,AI85FPRNet
from datasets.fpr_classification import fpr_get_datasets


#训练集
train_dataset,val_dataset = fpr_get_datasets()

x,y= train_dataset[0]
print(x.size(),y)
net = AI85FPRNet(num_classes=100).cuda()
# net = MobileNetV2(num_classes=100).cuda()
# y = net(torch.randn(1, 3, 32, 32))
# print(y.size())

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class AverageMeter(object):
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
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""  # [128, 10],128
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # [128, 5],indices

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 5,128

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
#超参数
warm=1
epoch=160
batch_size=128
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1) #learning rate decay

from torch.utils.data import  DataLoader
trainloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=5, pin_memory=True)
valloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=5, pin_memory=True)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)


def train(trainloader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 100-top1.avg

def validate(val_loader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

    # print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return 100-top1.avg
best_prec = 0
for e in range(epoch):
        train_scheduler.step( e)

        # train for one epoch
        train_acc=train(trainloader, net, loss_function, optimizer, e)

        # evaluate on test set
        test_acc = validate(valloader, net, loss_function)

        print('epoch:{}  train_acc:{:.3}  test_acc:{:.3} '.format(e,train_acc,test_acc))
        # remember best precision and save checkpoint
        is_best = test_acc > best_prec
        if is_best:
            print("saving best model!")
            torch.save(net.state_dict(), "student.pth")
        best_prec = max(test_acc,best_prec)
print(best_prec)