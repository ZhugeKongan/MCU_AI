import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats
from scipy import integrate
from sklearn.decomposition import PCA

def readdata(filepath):
    columnNames=['id','activity','timestamp','x_axis','y_axis','z_axis']
    data=pd.read_csv(filepath,header=None,names=columnNames,na_values=';')
    return data
def featurenomalize(data):
    mean=np.mean(data,axis=0)
    sigma=np.std(data,axis=0)
    return (data-mean)/sigma
# defining the function to plot a single axis data
def plotAxis(axis,x,y,title):
    axis.plot(x,y)#画点
    axis.set_title(title)
    axis.xaxis.set_visible(False)#不显示x轴
    axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])#y轴坐标范围
    axis.set_xlim([min(x), max(x)])#x轴坐标范围
    axis.grid(True)#打开栅格
def plotActivity(activity,data):
    fig,(ax0,ax1,ax2)=plt.subplots(nrows=3,figsize=(15,10),sharex=True)
    plotAxis(ax0,data['timestamp'],data['x_axis'],'x_axis')
    plotAxis(ax1, data['timestamp'], data['y_axis'], 'y_axis')
    plotAxis(ax2, data['timestamp'], data['z_axis'], 'z_axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()

def windows(data,size):
    start = 0
    while start< data.count():
        yield int(start), int(start + size)
        start+= (size/2)

def segment_signal(data,window_size=90):
    segments=np.empty((0,window_size,3))
    lables=np.empty((0))
    for (start,end) in windows(data['timestamp'],window_size):
        #print(start,end)
        x = data['x_axis'][start:end]
        y = data['y_axis'][start:end]
        z = data['z_axis'][start:end]
        if(len(data['timestamp'][start:end])==window_size):
            '''vtsack:vertical，实现轴0（行）合并，segments[0]=segments，segments[1]=np.dstack([x,y,z])
              hstack:horizontal，轴1（列）合并
              dstack：deep stack，轴2（高）合并，如果a、b是一维数组，或是二维数组，系统首先将a、b变为三维数组，再按照2号轴进行合并操作，把a追加到c中，再把b的元素排到c中。'''
            segments=np.vstack([segments,np.dstack([x,y,z])])
            #返回传入数组/矩阵中最常出现的成员以及出现的次数
            lables=np.append(lables,stats.mode(data['activity'][start:end])[0][0])
    return segments,lables

#加速度转速度，或速度转位置
def definite_integral(data,xv = np.empty((0)),yv = np.empty((0)),zv = np.empty((0))):
    for k in range(len(data['x_axis'])):#180
        #global xv
        global xtemp
        start = data['x_axis'].index[0]
        xtemp =ytemp=ztemp=0
        for i in range(k):
            x=start+i
            fx=data['x_axis'][x]
            #print(fx)
            fy = data['y_axis'][x]
            fz = data['z_axis'][x]
            t = data['timestamp'][x + 1] - data['timestamp'][x]
            t=t/10000000000
            xtemp+=(fx*t)
            ytemp += (fy * t)
            ztemp += (fz * t)
            #print(i,x,fx,t,xt,xtemp)
        #print(xtemp)
        xv=np.append(xv,xtemp)
        yv = np.append(yv, ytemp)
        zv = np.append(zv, ztemp)
    #print(xv)
    #print(xv.shape)
    xv[0]=xv[1]
    yv[0] = yv[1]
    zv[0] = zv[1]
    data['x_axis']=xv
    data['y_axis'] = yv
    data['z_axis'] = zv
    #print(data['x_axis'])
    return data
def draw_3DAcceleration(dataset):
    for activity in np.unique(dataset['activity']):  # 除去重复的标签
        subset = dataset[dataset['activity'] == activity][:180]
        plotActivity(activity, subset)

def draw_3Dspeed(dataset):
    for activity in np.unique(dataset['activity']):  # 除去重复的标签
        subset = dataset[dataset['activity'] == activity][:180]
        subset = definite_integral(subset)
        #print(subset)
        #print(subset.shape)
        plotActivity(activity,subset)

def draw_Displacement(dataset):
    for activity in np.unique(dataset['activity']):  # 除去重复的标签
        subset = dataset[dataset['activity'] == activity][:180]
        subset = definite_integral(subset)
        subset = definite_integral(subset)
        #print(subset)
        # print(subset.shape)
        # 生成画布
        figure = plt.figure()
        ax = figure.add_subplot(111, projection='3d')
        x=subset['x_axis']
        y=subset['y_axis']
        z=subset['z_axis']
        ax.plot(x, y,z)
        ax.set_xlim([min(x) - np.std(x), max(x) + np.std(x)])
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_zlim([min(z) - np.std(z), max(z) + np.std(z)])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.subplots_adjust(hspace=0.2)
        figure.suptitle(activity)
        plt.subplots_adjust(top=0.9)
        plt.show()

def PCA_reduceD(data):
    for activity in np.unique(data['activity']):  # 除去重复的标签
        subset = data[data['activity'] == activity][:180]
        subset = definite_integral(subset)
        subset = definite_integral(subset)
        X=np.vstack([subset['x_axis'],subset['y_axis'],subset['z_axis']])
        X=np.transpose(X)
        #print(X)
        #print(X.shape)

        # 使用sklearn的PCA进行维度转换
        # 建立PCA模型对象 n_components控制输出特征个数
        pca_model = PCA(n_components=2)#降为2维
        # 将数据集输入模型
        pca_model.fit(X)#训练
        # 对数据集进行转换映射
        newX=pca_model.transform(X)
        newX= np.transpose(newX)
        # print(newX[0])
        # print(newX.shape)
        # 获得转换后的所有主成分
        components = pca_model.components_
        # 获得各主成分的方差
        components_var = pca_model.explained_variance_
        # 获取主成分的方差占比
        components_var_ratio = pca_model.explained_variance_ratio_
        # 打印方差
        # print(np.round(components_var, 3))
        # # 打印方差占比
        # print(np.round(components_var_ratio, 3))


        #画图
        # 生成画布
        #figure = plt.figure()
        #ax = figure.add_subplot(111, projection='3d')
        figure, ax = plt.subplots(nrows=1)
        x = newX[0]
        y = newX[1]
        ax.plot(x, y)
        ax.set_xlim([min(x) - np.std(x), max(x) + np.std(x)])
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
        plt.subplots_adjust(hspace=0.2)
        figure.suptitle(activity)
        plt.subplots_adjust(top=0.9)
        plt.show()


