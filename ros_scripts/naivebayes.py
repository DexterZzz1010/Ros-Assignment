# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score


class NaiveBayes:
    def __init__(self):
        datafile = np.load('dataset/world1-2-5***v=008.npz')  # datefile: train train_label num_list
        self.train_data = datafile['train']  # 数据集
        self.train_labels = datafile['train_labels']  # 标签
        self.naive_bayes_test()
        # print("训练集大小： ", self.train_data)
        # print('标签： ', self.train_labels)
        # print("单个样本的大小", self.train_data[1].size)
        # print("训练集维度： ", self.train_data.ndim)

    def naive_bayes_test(self):
        x_train, x_test, y_train, y_test = train_test_split(self.train_data, self.train_labels,
                                                            test_size=0.3)  # 将数据集划分为训练集和测试集， 比例0.3
        GNB = GaussianNB()  # 采用高斯贝叶斯
        GNB.fit(x_train, y_train)
        score = GNB.score(x_test, y_test)
        print("准确度：", score)

        GNB.fit(self.train_data, self.train_labels)

        scores = cross_val_score(GNB, self.train_data, self.train_labels, cv=5,
                                 scoring='accuracy')  # 采用K折交叉验证的方法来验证算法效果
        print('K折准确度:', scores)

    # @staticmethod
    # def naive_bayes_decision(self, current_data):
    #     GNB = GaussianNB
    #     GNB.fit(self.train_data, self.train_labels)
    #     pre = GNB.predict(current_data)  # 用当前位置信息预测运动方向
    #     pre = pre[0]
    #     print(pre)
    #     return pre


# def naive_bayes_decision(current_data):
#
#     datafile = np.load('dataset/world1-2.npz')  # datefile: train train_label num_list
#     train_data = datafile['train']  # 数据集
#     train_labels = datafile['train_labels']  # 标签
#
#     GNB = GaussianNB()
#     GNB.fit(train_data,train_labels)
#     pre = GNB.predict(current_data)  # 用当前位置信息预测运动方向
#     pre = pre[0]
#     print(pre)
#     return pre


if __name__ == '__main__':
    NaiveBayes()
