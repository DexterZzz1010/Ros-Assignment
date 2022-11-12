import cv2
import cv2 as cv
import numpy as np
import os
import pandas as pd
import csv
import timeit
# import imutils
import json
import base64
from itertools import islice


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# def ShapeDetection(img):
#     img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓点
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     area = cv2.contourArea(contours[0])  # 计算轮廓内区域的面积
#     cv2.drawContours(imgContour, contours[0], -1, (255, 0, 0), 4)  # 绘制轮廓线
#     perimeter = cv2.arcLength(contours[0], True)  # 计算轮廓周长
#     approx = cv2.approxPolyDP(contours[0], 0.02 * perimeter, True)  # 获取轮廓角点坐标
#     CornerNum = len(approx)  # 轮廓角点的数量
#     x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度
#
#     # 轮廓对象分类
#     if CornerNum == 3:
#         objType = "triangle"
#     elif CornerNum == 4:
#         if w == h:
#             objType = "Square"
#         else:
#             objType = "Rectangle"
#     elif CornerNum > 4:
#         objType = "Circle"
#     else:
#         objType = "N"
#
#     cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边界框
#     cv2.putText(imgContour, objType, (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)  # 绘制文字


img = cv.imread("map.png")

img2 = img

readimg = img

img = cv.bilateralFilter(readimg, 21, 75, 75)
# img = cv.medianBlur(img, 3)
img = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变
img_1 = img.copy()
lower_blue = np.array([110, 150, 150])  # 浅蓝色
upper_blue = np.array([130, 255, 255])  # 深蓝色

mask_black = cv2.inRange(img, lower_blue, upper_blue)

# 轮廓检测

img, contours, _ = cv2.findContours(
    mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_SIMPLEX

contours = sorted(contours, key=cv2.contourArea, reverse=True)

rect = cv2.minAreaRect(contours[0])
# print(rect)
box = cv2.boxPoints(rect)
box = np.int0(box)
img2 = cv2.drawContours(img2, [box], 0, (0, 255, 0), 2)

pts1 = np.float32(box)
pts2 = np.float32([[rect[0][0] + rect[1][1] / 2, rect[0][1] + rect[1][0] / 2],
                   [rect[0][0] - rect[1][1] / 2, rect[0][1] + rect[1][0] / 2],
                   [rect[0][0] - rect[1][1] / 2, rect[0][1] - rect[1][0] / 2],
                   [rect[0][0] + rect[1][1] / 2, rect[0][1] - rect[1][0] / 2]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))

# 此处可以验证 box点的顺序
color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
i = 0
for point in pts2:
    cv2.circle(dst, tuple(point), 2, color[i], 4)
    i += 1

targets = dst[int(pts2[2][1]):int(pts2[1][1]), int(pts2[2][0]):int(pts2[3][0]), :]

box = sorted(box, key=lambda x: x[1])
line = sorted(box[-2:], key=lambda x: x[0])
r, x, y = np.sqrt(np.power(line[0] - line[1], 2).sum()), line[0][0], line[0][1]
# print(r, x, y)
Target=(x+(r/2),y)
print(Target)
cv2.imshow('img2', img2)
cv2.imshow('dst', dst)
cv2.imshow('target', targets)
cv2.waitKey()
cv2.destroyAllWindows()
