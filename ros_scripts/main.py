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
import math
from itertools import islice


def conversion(x, center):
    r = math.sqrt(math.pow(x[0] - center[0], 2) + math.pow(x[1] - center[1], 2))
    theta = math.atan2(x[1] - center[1], x[0] - center[0]) / math.pi
    if theta >= 0:
        theta = theta
    else:
        theta = 2 + theta
    result = [r, theta]
    return result


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def target(img, center):
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

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    rect = cv2.minAreaRect(contours[0])
    # print(rect)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # img2 = cv2.drawContours(img2, [box], 0, (0, 255, 0), 2)
    #
    # pts1 = np.float32(box)
    # pts2 = np.float32([[rect[0][0] + rect[1][1] / 2, rect[0][1] + rect[1][0] / 2],
    #                    [rect[0][0] - rect[1][1] / 2, rect[0][1] + rect[1][0] / 2],
    #                    [rect[0][0] - rect[1][1] / 2, rect[0][1] - rect[1][0] / 2],
    #                    [rect[0][0] + rect[1][1] / 2, rect[0][1] - rect[1][0] / 2]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))
    #
    # # 此处可以验证 box点的顺序
    # color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
    # i = 0
    # for point in pts2:
    #     cv2.circle(dst, tuple(point), 2, color[i], 4)
    #     i += 1
    #
    # targets = dst[int(pts2[2][1]):int(pts2[1][1]), int(pts2[2][0]):int(pts2[3][0]), :]

    box = sorted(box, key=lambda x: x[1])
    line = sorted(box[-2:], key=lambda x: x[0])
    r, x, y = np.sqrt(np.power(line[0] - line[1], 2).sum()), line[0][0], line[0][1]
    # print(r, x, y)
    Target = [x + (r / 2), y]
    Target = conversion(Target, center)
    return Target


def obstacle(img, center):
    img2 = img

    readimg = img

    img = cv.bilateralFilter(readimg, 21, 75, 75)
    # img = cv.medianBlur(img, 3)
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变
    img_1 = img.copy()
    lower_black = np.array([0, 0, 120])
    upper_black = np.array([0, 0, 130])

    mask_black = cv2.inRange(img, lower_black, upper_black)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # 轮廓检测
    img, contours, _ = cv2.findContours(
        mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # 筛选超过设定面积的轮廓
    img_contours = []
    filter_contours=[]
    for i in range(len(contours)):
        img_temp = np.zeros(img.shape, np.uint8)
        img_contours.append(img_temp)

        area = cv.contourArea(contours[i], False)
        if area > 1000:
            filter_contours.append(i)



    Obstacle = []
    Obstacle_Size = []
    for i in filter_contours:
        rect = cv2.minAreaRect(contours[i])
        obstcale = rect[0]
        obstacle_size = rect[1]
        obstacle = conversion(obstcale, center)
        Obstacle.extend(obstacle)
        Obstacle_Size.extend(obstacle_size)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     img2 = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)
    #     pts1 = np.float32(box)
    #     pts2 = np.float32([[rect[0][0] + rect[1][1] / 2, rect[0][1] + rect[1][0] / 2],
    #                        [rect[0][0] - rect[1][1] / 2, rect[0][1] + rect[1][0] / 2],
    #                        [rect[0][0] - rect[1][1] / 2, rect[0][1] - rect[1][0] / 2],
    #                        [rect[0][0] + rect[1][1] / 2, rect[0][1] - rect[1][0] / 2]])
    #     M = cv2.getPerspectiveTransform(pts1, pts2)
    #     dst = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))
    #
    #     # 此处可以验证 box点的顺序
    #     color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
    #     i = 0
    #     for point in pts2:
    #         cv2.circle(dst, tuple(point), 2, color[i], 4)
    #         i += 1
    #     target = dst[int(pts2[2][1]):int(pts2[1][1]), int(pts2[2][0]):int(pts2[3][0]), :]
    #     cv2.imshow('dst', dst)
    #     cv2.imshow('target', target)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # cv2.imshow('img2', img2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return Obstacle, Obstacle_Size


def circle(img):
    readimg = img
    gray = cv.cvtColor(readimg, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 21, 75, 75)
    gray = cv.medianBlur(gray, 3)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)
    sobely = cv.convertScaleAbs(sobely)
    sobelxy = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    edges = cv.Canny(sobelxy, 50, 150)

    # 霍夫圆变换
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 3000, param1=100, param2=30, minRadius=00, maxRadius=40)

    if np.any(circles != None):
        circles = np.uint16(np.around(circles))  # 取整
    else:
        circles = np.array([[[0, 0, 0]]])

    choose = circles[0, :]

    # for i in circles[0, :]:
    #     # 画出来圆的边界
    #     cv.circle(readimg, (i[0], i[1]), i[2], (0, 0, 255), 2)
    #     # 画出来圆心
    #     cv.circle(readimg, (i[0], i[1]), 2, (0, 255, 255), 3)
    #
    # cv.imshow("Circle", readimg)
    # cv.waitKey()
    # cv.destroyAllWindows()
    # print('choose', choose)
    r, x, y = (choose[0, 2]), (choose[0, 0]), (choose[0, 1])
    Center = [x, y]

    return r, Center


if __name__ == '__main__':
    img = cv.imread("map2.png")
    dataset = []

    # 返回的坐标值都是 r+theta
    r, circle_center = circle(img)
    Obstacles, Obstacles_size = obstacle(img, circle_center)
    Target = target(img, circle_center)


    # print('r=%s' % r)
    # print(type(r))
    #
    # print('center=%s' % circle_center)
    # print(type(circle_center))
    #
    # print('obstacle=%s' % Obstacles)
    # print(type(Obstacles))
    #
    # print('obstacles size=%s' % Obstacles_size)
    # print(type(Obstacles_size))
    #
    # print('target=%s' % Target)
    # print(type(Target))

    # dataset.extend([r.tolist()])
    dataset.extend(Obstacles)
    # dataset.extend(Obstacles_size)
    dataset.extend(Target)

    print('dataset=%s' % dataset)
