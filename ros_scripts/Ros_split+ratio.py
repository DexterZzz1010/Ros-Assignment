import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import copy as cp
import random
import cv2
import collections
from PIL import Image
import sys
import cv2
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()




def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
        # (x,y)二元组表示粘贴上图相对下图的起始位置
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))

    return new_image


# 切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 3):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 3):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list


# 显示
def show_images(image_list):
    for image in image_list:
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        cv2.imshow("img", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def image_white(image):
    dimensions = image.shape

    height = image.shape[0]
    width = image.shape[1]
    size = height * width
    img, cnts, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size_elements = 0
    for cnt in cnts:
        # cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
        size_elements += cv2.contourArea(cnt)
        # print(cv2.contourArea(cnt))
    white_area_ratio = size_elements / size
    white_area_ratio=round(white_area_ratio,4)
    return white_area_ratio



def image_red(image_list):
    red_area_ratio = []
    for image in image_list:
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        img = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变
        lower_red = np.array([140, 0, 100])
        upper_red = np.array([180, 10, 255])
        # lower_red = np.array([0, 43, 46])
        # upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(img, lower_red, upper_red)
        # cv_show("red",mask_red)
        ratio = image_white(mask_red)
        red_area_ratio.append(ratio)
    return red_area_ratio



def image_blue(image_list):
    blue_area_ratio = []
    for image in image_list:
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变
        lower_blue = np.array([110, 150, 150])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(img, lower_blue, upper_blue)
        # cv_show("blue",mask_blue)
        ratio = image_white(mask_blue)
        blue_area_ratio.append(ratio)
    return blue_area_ratio


def image_green(image_list):
    green_area_ratio = []
    for image in image_list:
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        mask_green = cv2.inRange(img, lower_green, upper_green)
        cv_show("blue", mask_green)
        ratio = image_white(mask_green)
        green_area_ratio.append(ratio)
    return green_area_ratio



if __name__ == '__main__':
    file_path = "chess2.png"
    image = Image.open(file_path)
    # image.show()
    image = fill_image(image)
    image_list = cut_image(image)
    # print(image_list)
    # show_images(image_list)
    red_area_ratio = image_red(image_list)
    blue_area_ratio = image_blue(image_list)
    green_area_ratio = image_green(image_list)
    print("red_area_ratio=%s",red_area_ratio)
    print("blue_area_ratio=%s",blue_area_ratio)
    print("green_area_ratio=%s",green_area_ratio)
    # cv_show("chess",image)
