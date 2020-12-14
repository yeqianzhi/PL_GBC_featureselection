
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import GBNRS



def load_datasets():
    # 获取数据集
    m = loadmat("F:\\partial_datasets\\BirdSong.mat")
    data = m["data"]
    partial_target = m["partial_target"].todense().T
    target = m["target"].todense().T

    # 归一化
    minMax = MinMaxScaler()  # Normalize data
    data = minMax.fit_transform(data)
    return data, partial_target, target

    for i in range(10):
        attributes = GBNRS.get_attribute_reduction(data[:, :])
        Attributes.append(attributes)    
    
    return data, partial_target, target









