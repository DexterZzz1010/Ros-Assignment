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
readimg=cv.imread('map.png')

gray = cv.cvtColor(readimg, cv.COLOR_BGR2GRAY)
gray = cv.bilateralFilter(gray, 21, 75, 75)
gray = cv.medianBlur(gray, 3)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
sobelx = cv.convertScaleAbs(sobelx)
sobely = cv.convertScaleAbs(sobely)
sobelxy = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
edges = cv.Canny(sobelxy, 50, 150)

# scharrx = cv.Scharr(gray, cv.CV_64F, 1, 0)
# scharry = cv.Scharr(gray, cv.CV_64F, 0, 1)
# scharrx = cv.convertScaleAbs(scharrx)
# scharry = cv.convertScaleAbs(scharry)
# scharrxy = cv.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
# edges = cv.Canny(scharrxy, 50, 150)


# laplacian = cv.convertScaleAbs(cv.Laplacian(gray, cv.CV_64F))
# edges = cv.Canny(laplacian, 50, 150)


# 霍夫圆变换
circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 3000, param1=100, param2=30, minRadius=00, maxRadius=40)

if np.any(circles != None):
    circles = np.uint16(np.around(circles))  # 取整
else:
    circles = np.array([[[0, 0, 0]]])

choose = circles[0, :]

for i in circles[0, :]:
    # 画出来圆的边界
    cv.circle(readimg, (i[0], i[1]), i[2], (0, 0, 255), 2)
    # 画出来圆心
    cv.circle(readimg, (i[0], i[1]), 2, (0, 255, 255), 3)

cv.imshow("Circle", readimg)
cv.waitKey()
cv.destroyAllWindows()
print('choose',choose)
# print('choose[]',choose[0, 2]), (choose[0, 0]), (choose[0, 1])
r, x, y = (choose[0, 2]), (choose[0, 0]), (choose[0, 1])
print(r,x,y)



