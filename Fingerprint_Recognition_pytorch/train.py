import os
import warnings
import functools
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

# import onnx
# #https://github.com/666DZY666/micronet
# from micronet.base_module.op import *
# import micronet.compression.quantization.wqaq.dorefa.quantize as quant_dorefa
# import micronet.compression.quantization.wqaq.iao.quantize as quant_iao


import torch
import torchvision
import torch.optim as optim
from torchsummary import summary
from torch import sigmoid,softmax
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms

from args import *
from fcv_dataloder import FVCDataset,FRDataset
from metrics import AverageMeter,accuracy


# from model.MobileNet import MobileNetV2
from model.MobileNetV2 import MobileNetV2,MobileNetV2Slim
# from model.ShuffleNet import shufflenet_g1
from model.ShuffleNetV2 import shufflenet_v2_x0_5

'''***********- Hyper Arguments-*************'''
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


print("***********- ***********- READ DATA and processing-*************")
normalize = transforms.Normalize(mean=[0.5],std=[0.5])
train_transforms = transforms.Compose([
            # transforms.CenterCrop(224),
            # transforms.RandomRotation(30),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(224),
            transforms.Resize(data_config.input_size),
            transforms.ToTensor(),
            normalize])
test_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(data_config.input_size),
            transforms.ToTensor(),
            normalize])
# train_dataset =FVCDataset(data_config.data_path+'img_train.npy',data_config.data_path+'label_train.npy')
# val_dataset = FVCDataset(data_config.data_path+'img_val.npy',data_config.data_path+'label_val.npy' )
# test_dataset = FVCDataset(data_config.data_path+'img_test.npy',data_config.data_path+'label_test.npy' )
# x1,y1= train_dataset[0]
# x2,y2= val_dataset[0]
# x3,y3= test_dataset[0]
# print(x1.size(),y1,x2.size(),y2,x3.size(),y3)
train_dataset =FRDataset(data_config.data_path+'train',train_transforms)
val_dataset = FRDataset(data_config.data_path+'test',test_transforms )
x,y= train_dataset[0]
print(x.size(),y)


print("***********- loading model -*************")
# model =MobileNetV2Slim(num_classes=data_config.num_class).cuda()
# model =shufflenet_g1(num_classes=data_config.num_class).cuda()
model =shufflenet_v2_x0_5(num_classes=data_config.num_class).cuda()
# model = torch.nn.DataParallel(model,device_ids=[0])
model_path='MobileNetV2.onnx'
# model.load_state_dict(torch.load(model_path))

optimizer = eval(data_config.optimizer)(model.parameters(),**data_config.optimizer_parm)
scheduler = eval(data_config.scheduler)(optimizer,**data_config.scheduler_parm)
loss_f=eval(data_config.loss_f)()
loss_dv=eval(data_config.loss_dv)()
loss_fn = eval(data_config.loss_fn)()


'''***********- trainer -*************'''
class trainer:
    def __init__(self, loss_f,loss_dv,loss_fn, model, optimizer, scheduler, config):
        self.loss_f = loss_f
        self.loss_dv = loss_dv
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def batch_train(self, batch_imgs, batch_labels, epoch):
        predicted = self.model(batch_imgs)
        loss = self.myloss(predicted, batch_labels)
        predicted = softmax(predicted, dim=-1)
        del batch_imgs, batch_labels
        return loss, predicted

    def train_epoch(self, loader,epoch):
        self.model.train()
        tqdm_loader = tqdm(loader)
        # acc = Accuracy_score()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("\n************Training*************")
        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            #print("data",imgs.size(), labels.size())#[128, 3, 32, 32]) torch.Size([128]
            imgs, labels=imgs.cuda(), labels.cuda()#permute(0,3,1,2).
            # print(self.optimizer.param_groups[0]['lr'])
            loss, predicted = self.batch_train(imgs, labels, epoch)
            losses.update(loss.item(), imgs.size(0))
            # print(predicted.size(),labels.size())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            err1, err5 = accuracy(predicted.data, labels, topk=(1, 5))
            top1.update(err1.item(), imgs.size(0))
            top5.update(err5.item(), imgs.size(0))

            tqdm_loader.set_description('Training: loss:{:.4}/{:.4} lr:{:.4} err1:{:.4} err5:{:.4}'.
                                        format(loss, losses.avg, self.optimizer.param_groups[0]['lr'],top1.avg, top5.avg))
            # if batch_idx%1==0:
            #     break
        return 100-top1.avg, losses.avg

    def valid_epoch(self, loader, epoch):
        self.model.eval()
        # acc = Accuracy_score()
        # tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()

        print("\n************Evaluation*************")
        for batch_idx, (imgs, labels) in enumerate(loader):
            with torch.no_grad():
                batch_imgs = imgs.cuda()#permute(0,3,1,2).
                batch_labels = labels.cuda()
                predicted= self.model(batch_imgs)
                loss = self.myloss(predicted, batch_labels).detach().cpu().numpy()
                loss = loss.mean()
                predicted = softmax(predicted, dim=-1)
                losses.update(loss.item(), imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), imgs.size(0))

        return 100-top1.avg, losses.avg



    def myloss(self,predicted,labels):
        #print(predicted.size(),labels.size())#[128, 10]) torch.Size([128])
        loss = self.loss_f(predicted,labels,)
        # loss = loss1+loss2
        return loss

    def run(self, train_loder, val_loder,model_path):
        best_acc = 0
        start_epoch=0
        # model, optimizer, start_epoch=load_checkpoint(self.model,self.optimizer,model_path)
        for e in range(self.config.epochs):
            e=e+start_epoch+1
            print("------model:{}----Epoch: {}--------".format(self.config.model_name,e))
            self.scheduler.step(e)
            # torch.cuda.empty_cache()

            train_acc, train_loss = self.train_epoch(train_loder,e)
            val_acc, val_loss=self.valid_epoch(val_loder,e)
            #
            print("\nval_loss:{:.4f} | val_acc:{:.4f} | train_acc:{:.4f}".format(val_loss, val_acc,train_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                print('Current Best (top-1 acc):',val_acc)
                # Export the model
                #import onnxoptimizer

                #new_model = onnxoptimizer.optimize(self.model)
                # new_model = quant_iao.prepare(
                #     self.model,
                #     inplace=False,
                #     a_bits=8,
                #     w_bits=8,
                #     q_type=0,
                #     q_level=0,
                #     weight_observer=0,
                #     bn_fuse=False,
                #     bn_fuse_calib=False,
                #     pretrained_model=False,
                #     qaft=False,
                #     ptq=False,
                #     percentile=0.9999,
                # )
                #new_model = quant_dorefa.prepare(self.model, inplace=False, a_bits=8, w_bits=8)
                x = torch.rand(1, 3,data_config.input_size, data_config.input_size).float().cuda()
                save_path=data_config.MODEL_PATH+data_config.model_name+'_epoch{}_params.onnx'.format(e)
                torch.onnx.export(self.model, x, save_path, export_params=True, verbose=False,opset_version=10)
                # 支持Opset 7, 8, 9 and 10 of ONNX 1.6 is supported.
                print("saving model sucessful !",save_path)
                # onnx_model = onnx.load(save_path)
                # onnx.checker.check_model(onnx_model)


        print('\nbest score:{}'.format(data_config.model_name))
        print("best accuracy:{:.4f}  ".format(best_acc))

# print('''***********- training -*************''')
Trainer = trainer(loss_f,loss_dv,loss_fn,model,optimizer,scheduler,config=data_config)
train = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=data_config.WORKERS, pin_memory=True,drop_last=True)
val = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.WORKERS, pin_memory=True,drop_last=True)
Trainer.run(train,val,model_path)




