
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import MinMaxScaler




def load_datasets():
    # 获取数据集
    dir = "F:\\partial_datasets\\BirdSong.mat"
    m = loadmat(dir)
    data_no_label = m["data"]
    # print(type(data_no_label))
    # print(m["partial_target"])
    partial_target = m["partial_target"].todense().T
    # partial_target = m["partial_target"].T
    
    target = m["target"].todense().T
    # target = m["target"].T

    num_data, num_attribute = data_no_label.shape

    # 归一化
    minMax = MinMaxScaler()  # Normalize data
    data_no_label = np.hstack((minMax.fit_transform(data_no_label), data_no_label[:, 0].reshape(num_data, 1)))
    return data_no_label, partial_target, target













