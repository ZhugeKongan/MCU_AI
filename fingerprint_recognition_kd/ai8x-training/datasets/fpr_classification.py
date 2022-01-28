#
#
# """
# figerprint-recognition Dataset
# """
# import os
# import numpy as np
# import torchvision
# from torchvision import transforms
# from PIL import Image
#
# import torch
# from torch.utils.data import Dataset
#
# class FVCDataset(Dataset):
#
#     def __init__(self, data_file, transform=None):
#         '''
#         data_file: 指纹的.npy格式数据
#         label_file:指纹的.npy格式标签
#         '''
#         # 所有图片的绝对路径
#         self.datas=os.listdir(data_file)
#         self.data_file=data_file
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img_path=self.datas[index]
#         data=Image.open(self.data_file+'/'+img_path)
#         label, _= img_path.split('_')
#         label=int(label)-1
#         # data=np.expand_dims(data,-1);
#         # data =np.repeat(data,3,-1);
#         # print(data.shape)
#         if self.transform is not None:
#             data = self.transform(data)
#         data=torch.repeat_interleave(data,3,0);
#
#         return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))
#
#     def __len__(self):
#         return len(self.datas)
#
# def fpr_get_datasets( load_train=True, load_test=True):
#     """
#     Load the figerprint-recognition dataset.
#     """
#     data_dir='/disks/disk2/lishengyan/dataset/fingerprint/'
#
#     if load_train:
#         data_path =data_dir+'train'
#
#         train_transform = transforms.Compose([
#             transforms.RandomCrop(224, padding=0),
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),#数据归一化
#         ])
#
#         train_dataset = FVCDataset(data_path,train_transform)
#     else:
#         train_dataset = None
#
#     if load_test:
#         data_path = data_dir + 'test'
#
#         test_transform = transforms.Compose([
#             transforms.RandomCrop(224, padding=0),
#             transforms.Resize(64),
#             #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),#数据归一化
#             transforms.ToTensor(),
#         ])
#
#         test_dataset = FVCDataset(data_path,test_transform)
#
#
#     else:
#         test_dataset = None
#
#     return train_dataset, test_dataset
#
#
# # datasets = [
# #     {
# #         'name': 'fpr',
# #         'input': (3, 64, 64),
# #         'output': ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8','F9', 'F10','F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18','F19', 'F20',
# #                    'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28','F29', 'F30','F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38','F39', 'F40',
# #                    'F41', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47', 'F48','F49', 'F50','F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57', 'F58','F59', 'F60',
# #                    'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F67', 'F68','F69', 'F70','F71', 'F72', 'F73', 'F74', 'F75', 'F76', 'F77', 'F78','F79', 'F80',
# #                    'F81', 'F82', 'F83', 'F84', 'F85', 'F86', 'F87', 'F88','F89', 'F90','F91', 'F92', 'F93', 'F94', 'F95', 'F96', 'F97', 'F98','F99', 'F100'
# #                    ),
# #         'loader': fpr_get_datasets,
# #     },
# # ]


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

    def __getitem__(self, index):
        img_path=self.datas[index]
        data = Image.open(self.data_file + '/' + img_path)
        label, _= img_path.split('_')
        label=int(label)-1

        if self.transform is not None:
            data = self.transform(data)
        data = torch.repeat_interleave(data, 3, 0);
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
        'output': ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8','F9', 'F10','F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18','F19', 'F20',
                   'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28','F29', 'F30','F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38','F39', 'F40',
                   'F41', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47', 'F48','F49', 'F50','F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57', 'F58','F59', 'F60',
                   'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F67', 'F68','F69', 'F70','F71', 'F72', 'F73', 'F74', 'F75', 'F76', 'F77', 'F78','F79', 'F80',
                   'F81', 'F82', 'F83', 'F84', 'F85', 'F86', 'F87', 'F88','F89', 'F90','F91', 'F92', 'F93', 'F94', 'F95', 'F96', 'F97', 'F98','F99', 'F100'
                   ),
        'loader': fpr_get_datasets,
    },
]
