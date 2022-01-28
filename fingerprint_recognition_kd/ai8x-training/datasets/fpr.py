

"""
figerprint-recognition Dataset
"""
import os
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image

import ai8x

import torch
from torch.utils.data import Dataset
from models.mobilenet_v2 import MobileNetV2


class FVCDataset(Dataset):

    def __init__(self, data_file, transform=None):
        '''
        data_file: 指纹的.npy格式数据
        label_file:指纹的.npy格式标签
        '''
        # 所有图片的绝对路径
        self.datas=os.listdir(data_file)
        self.data_file = data_file
        self.transform = transform
        # self.model = MobileNetV2(num_classes=100)#.cuda()
        # self.model.load_state_dict(torch.load("/disks/disk2/lishengyan/mcuev/max7800-dev/ai8x-training/teacher_model.pth"))
    def __getitem__(self, index):
        img_path=self.datas[index]
        data = Image.open(self.data_file + '/' + img_path)
        label, _= img_path.split('_')
        label=int(label)-1

        if self.transform is not None:
            data = self.transform(data)
        data = torch.repeat_interleave(data, 3, 0);
        # embedding=torch.unsqueeze(data,0)#.cuda()
        # with torch.no_grad():
        #     for name, module in self.model.named_children():
        #         #print(name)
        #         if name != 'dropout' and name !='linear':
        #             embedding = module(embedding)
        # embedding = torch.squeeze(embedding, 0)#.detach().cpu()
        return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.datas)

def fpr_get_datasets(data, load_train=True, load_test=True):
    """
    Load the figerprint-recognition dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://github.com/ZhugeKongan/Fingerprint-Recognition-pytorch-for-mcu).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    """
    (data_dir, args) = data
    data_dir='/disks/disk2/lishengyan/dataset/fingerprint/'

    if load_train:
        data_path =data_dir+'train'

        train_transform = transforms.Compose([
            transforms.RandomCrop(224, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(64),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = FVCDataset(data_path,train_transform)
    else:
        train_dataset = None

    if load_test:
        data_path = data_dir + 'test'

        test_transform = transforms.Compose([
            transforms.RandomCrop(224, padding=0),
            transforms.Resize(64),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = FVCDataset(data_path,test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'fpr',
        'input': (3, 64, 64),
        'output': ('id'),
        'regression': True,
        'loader': fpr_get_datasets,
    },
]
