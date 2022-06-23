import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FVCDataset(Dataset):

    def __init__(self, data_file,label_file, transform=None):
        '''
        data_file: 指纹的.npy格式数据
        label_file:指纹的.npy格式标签
        '''
        # 所有图片的绝对路径
        self.datas=np.load(data_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __getitem__(self, index):
        data=self.datas[index]
        label=self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        data=data.astype(np.float32) / 255.
        label = np.argmax(label)
        return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.datas)

class FRDataset(Dataset):

    def __init__(self, data_file, transform=None):
        '''
        data_file: 指纹的.npy格式数据
        label_file:指纹的.npy格式标签
        '''
        # 所有图片的绝对路径
        self.data_file=data_file
        self.datas=os.listdir(self.data_file)
        self.transform = transform

    def __getitem__(self, index):
        img_path=self.datas[index]
        data=Image.open(self.data_file+'/'+img_path)#260*260*1
        label, _= img_path.split('_')
        label=int(label)-1
        # print(data.shape, label)
        if self.transform is not None:
            data1 = self.transform(data)
            data2 = self.transform(data)
            data3 = self.transform(data)

        data=np.concatenate([data1,data2,data3],0)
        # print(data.size(), label)
        return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.datas)