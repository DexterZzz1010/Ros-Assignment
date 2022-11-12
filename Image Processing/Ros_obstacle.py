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
lower_grey = np.array([0, 0, 125])
upper_grey = np.array([0, 0, 150])

mask_black = cv2.inRange(img, lower_grey, upper_grey)

# 轮廓检测

img, contours, _ = cv2.findContours(
    mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_SIMPLEX

contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(contours)
list=[]

for i in range(0, 3):

    rect= cv2.minAreaRect(contours[i])
    list.append(rect)
    print(rect[1])
    print(type(rect[1]))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img2 = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)

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
    target = dst[int(pts2[2][1]):int(pts2[1][1]), int(pts2[2][0]):int(pts2[3][0]), :]
    cv2.imshow('dst', dst)
    cv2.imshow('target', target)
    cv2.waitKey()
    cv2.destroyAllWindows()


print(list)
cv2.imshow('img2', img2)
cv2.waitKey()
cv2.destroyAllWindows()
