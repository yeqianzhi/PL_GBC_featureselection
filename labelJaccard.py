'''
计算标记之间的Jaccard距离，得到标记之间的邻域
'''
from load_datasets import load_datasets
import numpy as np

data_no_label, partial_target, target = load_datasets()

# data_no_label = data_no_label[0:100, :]
# # print(data_no_label)
# partial_target = partial_target[0:100, :]


# 样本数、属性数、标记数
num_data, num_attribute = data_no_label.shape
num_target = partial_target.shape[1]
partial_target = partial_target.tolist()
# print('num_data:', num_data)


def Jaccard_neighbor(num_data, num_target, partial_target):

    '''
    将每个样本index对应的标记index存入label_index中
    '''
    label_index = {} # 存放每个样本index对应的标记index，都从0开始，eg: {0: [0, 2]}

    for i in range(num_data):
        label_index[i] = []
        # print(partial_target[i])
        for j in  range(num_target):
            # print('line 35',partial_target[i][j])
            if partial_target[i][j] == 1:
                if len(label_index[i]) == 0:
                    label_index[i] = [j]
                else:
                    label_index[i].append(j)
    # print('label_index:', label_index)    
    '''
    计算每个样本index在Jaccard距离下为0的近邻的index
    '''

    label_neighbor = {} # 存放每个样本index对应的k近邻index，都从0开始，eg: {0: [1, 3, 4, 5]}

    # 初试化dict
    for i in range(num_data):
        label_neighbor[i] = [i]

    # 计算Jaccard距离
    for i in range(num_data):
        S_i = label_index[i] # 第 i 个样本的候选标记集合
        # print('S_i:', S_i)
        temp_J_si_sj = [] # 暂时存放没有排序的所有距离
        for j in range(i + 1, num_data):
            S_j = label_index[j] # 第 j 个样本的候选标记集合
            # print('S_j:', S_j)

            # 交集
            mixed = [val for val in S_i if val in S_j]
            len_mixed = len(mixed)
            # 并集
            unioned = set(S_i + S_j)
            len_unioned = len(unioned)

            J_si_sj = (len_unioned - len_mixed)/ len_unioned # Si 与 Sj 的 Jaccard距离

            if J_si_sj == 0:
                label_neighbor[i].append(j)
                label_neighbor[j].append(i)

    
    return label_neighbor

    
# label_neighbor = Jaccard_neighbor(num_data, num_target, partial_target)
# print(label_neighbor)



