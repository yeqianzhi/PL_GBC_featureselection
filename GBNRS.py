
"""
granular ball neighborhood rough set
"""
import numpy as np
from collections import Counter
# from sklearn.cluster.k_means_ import k_means
from sklearn.cluster import KMeans
import warnings
import pandas as pd
import labelJaccard
import math
import copy

warnings.filterwarnings("ignore")

# pd.set_option('display.width', 5000)

class GranularBall:
    """class of the granular ball"""
    def __init__(self, data, num_target):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.num_target = num_target
        self.data = data[:, :]
        self.data_no_label = data[:, :-(num_target+1)]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.labels, self.purity = self.__get_label_and_purity()
        self.samples = []
        self.new_samples = []

    def __get_label_and_purity(self):
        """
        :return: the label and purity of the granular ball.
        """
        count = []
        for i in range(self.num_target):
            # res = np.sum(self.data[:, -(self.num_target+1):-1][:, i], axis = 0).tolist()
            res = np.sum(self.data[:, -(self.num_target+1):-1][:, i], axis = 0).tolist()[0][0]
            # print(res)
            count.append(res)
        max_sample = max(count)
        labels = []
        for i, val in enumerate(count):
            if val == max_sample:
                labels.append(i)
        purity = max_sample / self.num
        return labels, purity

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        data = self.data_no_label
        # label_cluster = k_means(X=self.data_no_label, n_clusters=2)[1]
        kmeans =  KMeans(n_clusters = 2, random_state = 0).fit(data)
        label_cluster = kmeans.predict(data)
        # print('label_cluster, split_2balls:', label_cluster)
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :], self.num_target)
            ball2 = GranularBall(self.data[label_cluster == 1, :], self.num_target)
        else:
            ball1 = GranularBall(self.data[0:1, :], self.num_target)
            ball2 = GranularBall(self.data[1:, :], self.num_target)
        return ball1, ball2


class GBList:
    """class of the list of granular ball"""
    def __init__(self, data, num_target):
        self.data = data[:, :]
        self.num_target = num_target
        self.granular_balls = [GranularBall(self.data, self.num_target)]  # gbs is initialized with all data


    def init_granular_balls(self, purity=1.0, min_sample=1):
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)
        i = 0
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        # self.granular_balls
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def del_balls(self, purity=0, num_data=0):
        """
        Deleting the balls that meets following conditions from the list, updating self.granular_balls and self.data.
        :param purity: delete the balls that purity is large than this value.
        :param num_data: delete the balls that the number of samples is large than this value.
        :return: None
        """
        self.granular_balls = [ball for ball in self.granular_balls if ball.purity >= purity and ball.num >= num_data]
        self.data = self.get_data()

    def re_k_means(self):
        """
        Global k-means clustering for data with the center of the ball as the initial center point.
        """
        k = len(self.granular_balls)
        data = self.data[:, :-(self.num_target+1)]
        # get_init = self.get_center()
        get_init = self.get_center()[:, 0, :]
        kmeans = KMeans(n_clusters = k, random_state = 0, init=get_init).fit(data)
        label_cluster = kmeans.predict(data)

        # 粒球划分
        granular_cluster = {}
        for i in range(k):
            granular_cluster[i] = []
        for i in range(len(label_cluster)):
            index = label_cluster[i]
            granular_cluster[index].append(i)

        for i in range(k):
            self.granular_balls[i] = GranularBall(self.data[label_cluster == i, :], self.num_target)
            self.granular_balls[i].samples = granular_cluster.get(i)

    
    def re_division(self, i):
        """
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - self.num_target - 1))
        attributes.remove(i)
        data = self.data[:, attributes]
        # get_init = self.get_center()[:, attributes]
        get_init = self.get_center()[:, 0, :][:, attributes]
        kmeans = KMeans(n_clusters = k, random_state = 0, init=get_init).fit(data)
        label_cluster = kmeans.predict(data)

        # 粒球划分
        granular_cluster = {}
        for i in range(k):
            granular_cluster[i] = []
        for i in range(len(label_cluster)):
            index = label_cluster[i]
            granular_cluster[index].append(i)
        # print(granular_cluster)

        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :], self.num_target))
            self.granular_balls[i].new_samples = granular_cluster.get(i)

        return granular_balls_division


def get_attribute_reduction(data, num_target, partial_target):
    """
    The main function of attribute reduction.
    :param data: data set
    :return: reduced attribute set
    """
    
    num, dim = data[:, :-1].shape
    index = np.array(range(num)).reshape(num, 1)  # column of index
    data = np.hstack((data, index))  # Add the index column to the last column of the data

    # step 1.
    granular_balls = GBList(data, num_target)  # create the list of granular balls
    granular_balls.init_granular_balls()  # initialize the list
    granular_balls.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample

    # step 2.
    granular_balls.re_k_means()  # Global k-means clustering as fine tuning.
    granular_balls.del_balls(purity=1)  # delete the ball wh sample

    feature_neighbor = {}
    for granular_cluster in granular_balls.granular_balls:
        feature_neighbor = get_feature_neighbor(feature_neighbor, granular_cluster.samples)

    # 根据特征划分的邻域
    feature_neighbor = sortedDict(feature_neighbor)
    # print(feature_neighbor)

    # 根据标记划分的邻域
    label_neighbor = labelJaccard.Jaccard_neighbor(num, num_target, partial_target.tolist())
    
    # print(len(feature_neighbor))
    # print(len(label_neighbor))

    # 在粒球中的样本总数
    # num_granular_sample = len(feature_neighbor)
    # print(feature_neighbor.keys())

    H_D_red = get_H_D_red(num, feature_neighbor, label_neighbor)
    # print(H_D_red)

    # step 3.
    attributes_reduction = list(range(data.shape[1] - num_target - 1))
    
    # for i in range(data.shape[1] - num_target - 1):
    #     if len(attributes_reduction) <= 1:
    #         break

    start = 1
    while start:   
        SIG = []
        for i in range(len(attributes_reduction)):
            # 删除一个属性
            temp_attributes_reduction = copy.deepcopy(attributes_reduction)
            the_remove_i = attributes_reduction.index(i)
            temp_attributes_reduction.remove(i)  # remove the ith attribute
            # 重新进行粒球划分
            gb_division = granular_balls.re_division(the_remove_i)  # divide the data with center of granular balls            
            # 再计算邻域判别指数
            feature_neighbor = {}
            for granular_cluster in granular_balls.granular_balls:
                feature_neighbor = get_feature_neighbor(feature_neighbor, granular_cluster.new_samples)            
            H_D_red_del = get_H_D_red(num, feature_neighbor, label_neighbor)

            # 计算SIG
            temp_SIG = H_D_red_del - H_D_red
            SIG.append(temp_SIG)
        
        print(SIG)
        # 返回最小值的index
        index = SIG.index(min(SIG))
        print(SIG[index])
        # [0.1, 0.6]
        # 如果满足条件，则删除该属性
        if SIG[index] > 0.1:
            the_remove_i = attributes_reduction.index(i)
            attributes_reduction.remove(i)  # remove the ith attribute

            # step 1.
            granular_balls = GBList(np.hstack((data[:, attributes_reduction], data[:, -(num_target+1):])), num_target)
            granular_balls.init_granular_balls()
            granular_balls.del_balls(num_data=2)

            # step 2.
            granular_balls.re_k_means()
            granular_balls.del_balls(purity=1)            
        else:
            start = 0

        

        # 
            # print('H_D_red_del', H_D_red_del)

        # purity = [round(ball.purity, 3) for ball in gb_division]  # get the purity of the divided granular balls
        # print('purity', purity)
        # print('sum(purity)', sum(purity))
        # print('len(purity)', len(purity), end='\n')

        # if sum(purity) == (len(purity)):  # if the ith attribute can be reduced
        #     # Recreate the new list granular balls with attributes after the reduction
        #     # print('del_balls')
        #     # step 1.
        #     granular_balls = GBList(np.hstack((data[:, attributes_reduction], data[:, -(num_target+1):])), num_target)
        #     granular_balls.init_granular_balls()
        #     granular_balls.del_balls(num_data=2)

        #     # step 2.
        #     granular_balls.re_k_means()
        #     granular_balls.del_balls(purity=1)

        # else:  # If the ith attribute is can't be reduced, then add it back.
        #     attributes_reduction.append(i)
        #     attributes_reduction.sort()
    attributes_reduction = attributes_reduction
    return attributes_reduction


        
def get_feature_neighbor(feature_neighbor, sample):
    for val in sample:
        feature_neighbor[val] = sample
    return feature_neighbor

def sortedDict(feature_neighbor):
    dict = {}
    for i in sorted (feature_neighbor) : 
        dict[i] = feature_neighbor[i]
    return dict

def get_H_D_red(num, feature_neighbor, label_neighbor):
    num_square = num * num
    sum_mixed = 0
    sum_red = 0
    for key in feature_neighbor.keys():
        mixed = [val for val in feature_neighbor[key] if val in label_neighbor[key]]
        len_mixed = len(mixed)
        len_red = len(feature_neighbor[key])
        sum_mixed += len_mixed
        sum_red += len_red

    H_Dred = math.log(num_square/sum_mixed)
    H_red = math.log(num_square/sum_red)
    H_D_red = H_Dred - H_red
    return  H_D_red