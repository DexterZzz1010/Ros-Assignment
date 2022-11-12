# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
from sklearn import tree  # 导入决策树
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# from sklearn.neural_network import MLPClassifier

datafile = np.load('dataset/labeled_img_data_1654521003.npz')  # datefile: train train_label num_list
train_data = datafile['train']  # 数据集
train_labels = datafile['train_labels']  # 标签


def naive_bayes_decision(current_data):
    GNB = GaussianNB()
    GNB.fit(train_data, train_labels)
    pre = GNB.predict(current_data)  # 用当前位置信息预测运动方向
    return pre[0]


def decisiontree_decision(current_data):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_labels)
    pre = clf.predict(current_data)
    return pre[0]


def randomforest_decision(current_data):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc.fit(train_data, train_labels)
    pre = rfc.predict(current_data)
    return pre[0]


def voting_decision(current_data):
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
    return pre[0]


def knn_decision(current_data):
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels)
    pre = knn.predict(current_data)
    return pre[0]


def svm_decision(current_data):
    svm = SVC(decision_function_shape='ovo')
    svm.fit(train_data, train_labels)
    pre = svm.predict(current_data)
    return pre[0]

# def mlp_decision(current_data):
#     mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 20), max_iter=500)
#     mlp.fit(train_data, train_labels)
#     pre = mlp.predict(current_data)
#     return pre[0]
