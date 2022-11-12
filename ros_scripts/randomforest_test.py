# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
from sklearn import tree  # 导入决策树
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time


class RandomForests:
    '''
    labeled_img_data_1654146302.npz 准确度81.8
    '''

    def __init__(self):
        datafile = np.load('dataset/world1-2-1.npz')  # datefile: train train_label num_list
        self.train_data = datafile['train']  # 数据集
        self.train_labels = datafile['train_labels']  # 标签
        print(self.train_data)
        self.randomforests_test()

    def randomforests_test(self):
        x_train, x_test, y_train, y_test = train_test_split(self.train_data, self.train_labels,
                                                            test_size=0.3)  # 将数据集划分为训练集和测试集， 比例0.3

        rfc = RandomForestClassifier(n_estimators=20)
        rfc_s = cross_val_score(rfc, self.train_data, self.train_labels, cv=10)  # k折检验

        clf = tree.DecisionTreeClassifier()
        clf_s = cross_val_score(clf, self.train_data, self.train_labels, cv=10)

        plt.plot(range(1, 11), rfc_s, label="RandomForest")
        plt.plot(range(1, 11), clf_s, label="Decision Tree")
        plt.legend()
        plt.show()  # 输出表格对比

        clf = clf.fit(x_train, y_train)
        rfc = rfc.fit(x_train, y_train)
        score_c = clf.score(x_test, y_test)
        score_r = rfc.score(x_test, y_test)
        print("Single Tree:{}".format(score_c)
              , "Random Forest:{}".format(score_r)
              )

        scores = []
        times = []
        for i in range(1, 50):
            rfc = RandomForestClassifier(n_estimators=i)
            time1 = time.time()
            rfc = rfc.fit(x_train, y_train)
            score = rfc.score(x_test, y_test)
            time2 = time.time() - time1
            scores.append(score)
            times.append(time2)
        plt.plot(range(1, 50), scores, label="RandomForest")
        plt.legend()
        plt.show()  # 输出表格对比

        plt.plot(range(1, 50), times, label="RandomForest time")
        plt.legend()
        plt.show()  # 输出表格对比


if __name__ == '__main__':
    RandomForests()
