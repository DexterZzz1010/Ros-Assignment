#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class SVM:

    def __init__(self):
        datafile = np.load('dataset/world1-2-5***v=008.npz')  # datefile: train train_label num_list
        self.train_data = datafile['train']  # 数据集
        self.train_labels = datafile['train_labels']  # 标签
        print(self.train_data)
        self.SVM_test()

    def SVM_test(self):
        x_train, x_test, y_train, y_test = train_test_split(self.train_data, self.train_labels,
                                                            test_size=0.3)  # 将数据集划分为训练集和测试集， 比例0.3
        svm = SVC(decision_function_shape='ovo')
        svm.fit(x_train, y_train)
        score = svm.score(x_test, y_test)
        print("准确度：", score)

        svm.fit(self.train_data, self.train_labels)

        scores = cross_val_score(svm, self.train_data, self.train_labels, cv=5,
                                 scoring='accuracy')  # 采用K折交叉验证的方法来验证算法效果
        print('K折准确度:', scores)


if __name__ == '__main__':
    SVM()