
import numpy as np

xfile="/disks/disk2/lishengyan/mcuev/HAR-CNN-TensorFlow/testData1.npy"
yfile="/disks/disk2/lishengyan/mcuev/HAR-CNN-TensorFlow/groundTruth1.npy"
testx="/disks/disk2/lishengyan/mcuev/HAR-CNN-TensorFlow/testx_cnn.npy"
testy="/disks/disk2/lishengyan/mcuev/HAR-CNN-TensorFlow/testy_cnn.npy"

x=np.load(xfile)
y=np.load(yfile)
print(x.shape)
print(y.shape)
tx=x[:500]
ty=y[:500]
print(tx.shape)
print(ty.shape)
np.save(testx,tx)
np.save(testy,ty)