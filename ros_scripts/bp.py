#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分数据集

print(tf.__version__)
print(keras.__version__)

# 加载数据集
datafile = np.load('dataset/labeled_img_data_1654517495.npz')  # datefile: train train_label num_list
train_data = datafile['train']  # 数据集
train_labels = datafile['train_labels']  # 标签
X_train_full, x_test, y_train_full, y_test = train_test_split(train_data, train_labels,
                                                              test_size=0.3)  # 将数据集划分为训练集和测试集， 比例0.3
# 查看训练集的形状和数据类型
print(X_train_full.shape, X_train_full.dtype)
# 比例缩放和像素强度降低到0-1,创建一个验证集
X_valid, X_train = X_train_full[:20], X_train_full[20:]
Y_valid, Y_train = y_train_full[:20], y_train_full[20:]

# 搭建网络模型
model = tf.keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[300, 1]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(3, activation="softmax"))  # 输出十个概率分布，看属于哪一个
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))

model.summary()
# print(history)

# 绘制history曲线
pd.DataFrame(history.history).plot(figsize=(8, 20))
plt.grid(True)
plt.gca().set_ylim(0, 1.1)
plt.show()

# 在测试集上测试
print(model.evaluate(x_test, y_test))
# 仅使用测试集的前三个例子
X_new = x_test[:3]
Y_proba = model.predict(X_new)
print('Y_proba: %f', Y_proba.round(2))
y_pre = model.predict(X_new)
print(y_pre)
