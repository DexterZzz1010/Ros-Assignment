# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
import cv2
import rospy
import time
from robot import Robot
from PIL import Image
from geometry_msgs.msg import Twist
import classifier
from imgsplit import datareturn

rate = 10  # 向ros发布命令频率
v = 0.8  # 前进距离
angular = 0.8  # 旋转角度


def decision(img):
    """
    调用不同的机器学习方法进行决策
    naive_bayes_decision： 朴素贝叶斯
    decisiontree_decision： 决策树
    randomforest_decision： 随机森林
    voting_decision： 投票算法
    mlp_decision：多层感知器
    :param img: 输入图像
    :return: 决策出的行驶方向
    """
    data = datareturn(img)
    # direction = classifier.naive_bayes_decision(data)
    # direction = classifier.decisiontree_decision(data)
    direction = classifier.randomforest_decision(data)
    # direction = classifier.voting_decision(data)
    # direction = classifier.mlp_decision(data)
    # direction = classifier.svm_decision(data)
    return direction


class RobotNavigation:
    def __init__(self):
        self.robot = Robot()
        # 发布话题相关参数
        self.rate_run = rospy.Rate(rate)
        self.twist = Twist()
        self.control()  # 开始控制小车

    def control(self):
        print("prepare to navigation.............")
        frame = self.robot.get_image()

        while True:
            frame = self.robot.get_image()
            cv2.imshow("control img", frame)
            # command = cv2.waitKey(100) & 0xFF # 等待输入按键取后八位
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 图像数据类型转换
            command = decision(frame)
            print("get key:", command)

            # forward -- 0
            if command == 0:
                self.twist.linear.x = v
                self.twist.angular.z = 0
                print("前进")

            # forward-left -- 1
            elif command == 1:
                self.twist.linear.x = v
                self.twist.angular.z = angular
                print("左转")

            # forward-right -- 2
            elif command == 2:
                self.twist.linear.x = v
                self.twist.angular.z = -angular
                print("右转")

            # 向机器人底盘发布数据
            self.robot.publish_twist(self.twist)
            self.rate_run.sleep()


if __name__ == '__main__':
    try:
        RobotNavigation()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        pass
