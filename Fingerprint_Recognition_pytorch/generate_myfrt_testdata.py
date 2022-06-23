import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.5],std=[0.5])
test_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(128),
            transforms.ToTensor(),
            normalize])

def prepare_eval_data(data_file, transform=None):

    datas = os.listdir(data_file)
    imgs=[]
    labels=[]
    for  img_path in datas:
        data = Image.open(data_file + '/' + img_path)  # 260*260*1
        label, _ = img_path.split('_')
        label = int(label) - 1
        label_ohot=np.zeros(100)
        label_ohot[label]=1
        # print(data.shape, label)

        data1 = transform(data)
        data2 = transform(data)
        data3 = transform(data)

        data = np.concatenate([data1, data2, data3], 0)

        labels.append(label_ohot)
        imgs.append(data)
    imgs = np.array(imgs)
    labels = np.array(labels)
    print(imgs.shape,labels.shape)
    return imgs,labels








if __name__ == '__main__':
    data_path = '/disks/disk2/lishengyan/dataset/fingerprint/'

    testx,testy=prepare_eval_data(data_path+'test',test_transforms )
    print(testx.shape,testy.shape)
    print(testy[0])

    np.save('fpr100*5_testx_128.npy', testx)
    np.save('fpr100*5_testy_128.npy', testy)


