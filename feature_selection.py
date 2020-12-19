
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import GBNRS
from load_datasets import load_datasets
from collections import Counter


data_no_label, partial_target, target = load_datasets()

# data_no_label = data_no_label[0:100, :]
# # print(data_no_label)
# partial_target = partial_target[0:100, :]
# # print(partial_target)

# 样本数、属性数、标记数
num_data, num_attribute = data_no_label.shape
num_target = partial_target.shape[1]
# print(type(num_target))

# 拼接data和num_data num_target
data = np.hstack((data_no_label, partial_target))

# 传的参数 data num_target
attributes = GBNRS.get_attribute_reduction(data, num_target, partial_target)
print(attributes)



# # 测试Couter
# count = []
# for i in range(num_target):
#     res = np.sum(partial_target[:, i], axis = 0).tolist()[0][0]
#     print(res)
#     count.append(res)
# max_sample = max(count)
# labels = []
# for i, val in enumerate(count):
#     if val == max_sample:
#         labels.append(i)
# purity = max_sample / num_data
# print(labels, purity)

# instance_label_only = {}

# print((partial_target[1]))

# 找出单标记样本(单标记样本的index以及label)
# for i in range(num_data):
#     par_label_sum = np.sum(partial_target[i])
#     if par_label_sum == 1:
#         j = np.array(partial_target[i]).tolist()[0].index(1.0)
#         instance_label_only[i] = j
# print(instance_label_only)

# Attributes = []
# # 进行10次的属性约简？
# for i in range(10):
#     attributes = GBNRS.get_attribute_reduction(data)
#     Attributes.append(attributes)

