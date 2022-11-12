# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

datafile = np.load('dataset/world1-2-5***v=008.npz')  # datefile: train train_label num_list
train_data = datafile['train']  # 数据集
train_labels = datafile['train_labels']  # 标签
# random_state 相当于随机数种子random.seed() ，其作用是相同的。
# 因为同一算法模型在不同的训练集和测试集的会得到不同的准确率，无法调参。
# 所以在sklearn 中可以通过添加random_state，通过固定random_state的值，每次可以分割得到同样训练集和测试集。
# 因此random_state参数主要是为了保证每次都分割一样的训练集和测试机，大小可以是任意一个整数，在调参缓解，只要保证其值一致即可。
clf1 = SVC(decision_function_shape='ovo')
clf2 = RandomForestClassifier(n_estimators=20)
clf3 = MLPClassifier(hidden_layer_sizes=(100, 500, 20), max_iter=10000)
# 将上面三个基模型集成
eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['SVC', 'Random Forest', 'MLP', 'Ensemble']):
    # 参数scoring：accuracy cv：5 将数据集分为大小相同的5份，四份训练，一份测试
    # cross_val_score训练模型打分函数
    scores = cross_val_score(clf, train_data, train_labels, scoring='accuracy', cv=5)
    # scores.mean()分数、scores.std()误差
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))