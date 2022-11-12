# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


def voting_decision(current_data):
    datafile = np.load('dataset/world1-2.npz')  # datefile: train train_label num_list
    train_data = datafile['train']  # 数据集
    train_labels = datafile['train_labels']  # 标签
    # random_state 相当于随机数种子random.seed() ，其作用是相同的。
    # 因为同一算法模型在不同的训练集和测试集的会得到不同的准确率，无法调参。
    # 所以在sklearn 中可以通过添加random_state，通过固定random_state的值，每次可以分割得到同样训练集和测试集。
    # 因此random_state参数主要是为了保证每次都分割一样的训练集和测试机，大小可以是任意一个整数，在调参缓解，只要保证其值一致即可。
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    # 将上面三个基模型集成
    eclf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='hard')

    eclf.fit(train_data, train_labels)
    pre = eclf.predict(current_data)  # 用当前位置信息预测运动方向
    pre = pre[0]
    print(pre)
    return pre
