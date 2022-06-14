from util import *

# random_seed = 611
# np.random.seed(random_seed)
# matplotlib inline
# 使用自带的样式进行美化
plt.style.use('ggplot')

filepath=r"E:\Learing_materials\MCU-AI\STM32CUBE.AI\HAR-CNN-Keras-master\HAR-CNN-Keras-master\HAR-CNN-TensorFlow\actitracker_raw.txt"
dataset=readdata(filepath)
# # 各维度加速度图像
draw_3DAcceleration(dataset)
# # 各维度速度图像
#draw_3Dspeed(dataset)
# #画位移图像
draw_Displacement(dataset)
PCA_reduceD(dataset)