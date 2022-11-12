import gtk.gdk
import cv2
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import win32api
import win32gui
import win32con
from PIL import ImageGrab
import time
import random

# # 窗体标题  用于定位游戏窗体
# WINDOW_TITLE = "连连看"
# # 时间间隔随机生成 [MIN,MAX]
# TIME_INTERVAL_MAX = 0.06
# TIME_INTERVAL_MIN = 0.1
# # 游戏区域距离顶点的x偏移
# MARGIN_LEFT = 10
# # 游戏区域距离顶点的y偏移
# MARGIN_HEIGHT = 180
# # 横向的方块数量
# H_NUM = 19
# # 纵向的方块数量
# V_NUM = 11
# # 方块宽度
# POINT_WIDTH = 31
# # 方块高度
# POINT_HEIGHT = 35
# # 空图像编号
# EMPTY_ID = 0
# # 切片处理时候的左上、右下坐标：
# SUB_LT_X = 8
# SUB_LT_Y = 8
# SUB_RB_X = 27
# SUB_RB_Y = 27
# # 游戏的最多消除次数
# MAX_ROUND = 200











w = gtk.gdk.get_default_root_window()
sz = w.get_size() #窗口大小
print "The size of the window is %d x %d" % sz
pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0,sz[0],sz[1])
if (pb != None):
    pb.save("screenshot.png","png")
    print "Screenshot saved to screenshot.png."
else:
    print "Unable to get the screenshot."














# VideoCapture方法是cv2库提供的读取视频方法
cap = cv2.VideoCapture('C:\\Users\\xxx\\Desktop\\sweet.mp4')
# 设置需要保存视频的格式“xvid”
# 该参数是MPEG-4编码类型，文件名后缀为.avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 设置视频帧频
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置视频大小
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# VideoWriter方法是cv2库提供的保存视频方法
# 按照设置的格式来out输出
out = cv2.VideoWriter('C:\\Users\\xxx\\Desktop\\out.avi',fourcc ,fps, size)

# 确定视频打开并循环读取
while(cap.isOpened()):
    # 逐帧读取，ret返回布尔值
    # 参数ret为True 或者False,代表有没有读取到图片
    # frame表示截取到一帧的图片
    ret, frame = cap.read()
    if ret == True:
        # 垂直翻转矩阵
        frame = cv2.flip(frame,0)

        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
out.release()
# 关闭窗口
cv2.destroyAllWindows()


def getGameWindow():
    # FindWindow(lpClassName=None, lpWindowName=None)  窗口类名 窗口标题名
    window = win32gui.FindWindow(None, WINDOW_TITLE)

    # 没有定位到游戏窗体
    while not window:
        print('Failed to locate the game window , reposition the game window after 10 seconds...')
        time.sleep(10)
        window = win32gui.FindWindow(None, WINDOW_TITLE)

    # 定位到游戏窗体
    # 置顶游戏窗口
    win32gui.SetForegroundWindow(window)
    pos = win32gui.GetWindowRect(window)
    print("Game windows at " + str(pos))
    return (pos[0], pos[1])

def getScreenImage():
    print('Shot screen...')
    # 获取屏幕截图 Image类型对象
    scim = ImageGrab.grab()
    scim.save('screen.png')
    # 用opencv读取屏幕截图
    # 获取ndarray
    return cv2.imread("screen.png")

def getAllSquare(screen_image, game_pos):
    print('Processing pictures...')
    # 通过游戏窗体定位
    # 加上偏移量获取游戏区域
    game_x = game_pos[0] + MARGIN_LEFT
    game_y = game_pos[1] + MARGIN_HEIGHT

    # 从游戏区域左上开始
    # 把图像按照具体大小切割成相同的小块
    # 切割标准是按照小块的横纵坐标
    all_square = []
    for x in range(0, H_NUM):
        for y in range(0, V_NUM):
            # ndarray的切片方法 ： [纵坐标起始位置：纵坐标结束为止，横坐标起始位置：横坐标结束位置]
            square = screen_image[game_y + y * POINT_HEIGHT:game_y + (y + 1) * POINT_HEIGHT,
                     game_x + x * POINT_WIDTH:game_x + (x + 1) * POINT_WIDTH]
            all_square.append(square)

    # 因为有些图片的边缘会造成干扰，所以统一把图片往内缩小一圈
    # 对所有的方块进行处理 ，去掉边缘一圈后返回
    finalresult = []
    for square in all_square:
        s = square[SUB_LT_Y:SUB_RB_Y, SUB_LT_X:SUB_RB_X]
        finalresult.append(s)
    return finalresult

if __name__ == '__main__':
    random.seed()
    # i. 定位游戏窗体
    game_pos = getGameWindow()
    time.sleep(1)
    # ii. 获取屏幕截图
    screen_image = getScreenImage()

