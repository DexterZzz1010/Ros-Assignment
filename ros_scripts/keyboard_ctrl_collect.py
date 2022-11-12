# encoding: utf-8
__author__ = 'Gary_Zhang'

import numpy as np
import cv2
import rospy
import time
from PIL import Image
import imgsplit
from robot import Robot
from geometry_msgs.msg import Twist
import os

rate = 10  # 向ros发布命令频率
v = 0.08  # 前进距离
angular = 0.1  # 旋转角度


class CollectTrainingData:
    def __init__(self):
        # 控制底盘
        self.robot = Robot()

        # 发布话题相关参数
        self.rate_run = rospy.Rate(rate)
        self.twist = Twist()

        """
        cmd:
        w:前进
        a:左转
        d:右转
        s:后退
        i:左自转
        j:右自转
        l:停止
        """

        self.data_path = "dataset"
        self.saved_file_name = 'labeled_img_data_' + str(int(time.time()))
        self.collect_data()  # 开始控制小车

    def collect_data(self):
        label = []
        num_list = [0, 0, 0, 0, 0]
        data = []
        total_images_collected = 0

        print(".............prepare to collect.............")

        while True:
            frame = self.robot.get_image()
            print(type(frame))
            cv2.imshow("control img", frame)
            command = cv2.waitKey(100) & 0xFF  # 等待输入按键取后八位
            # command = input("enter command:")
            print("get key:", command)
            if command == ord('q'):
                print("--------------quit---------------")
                break

            elif command == 255:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                continue

            elif command == ord('w'):
                self.twist.linear.x = v
                self.twist.angular.z = 0
                print('前进')

            elif command == ord('a'):
                self.twist.linear.x = v
                self.twist.angular.z = angular
                print("左转")

            # forward-right -- 2
            elif command == ord('d'):
                self.twist.linear.x = v
                self.twist.angular.z = -angular
                print("右转")


            # stop-sign -- 3
            elif command == ord('s'):
                self.twist.linear.x = -v
                self.twist.angular.z = 0
                print("后退")


            # road banner front
            elif command == ord('i'):
                self.twist.linear.x = 0.1 * v
                self.twist.angular.z = angular
                print("左自转")
                continue

            # road banner left
            elif command == ord('j'):
                self.twist.linear.x = 0.1 * v
                self.twist.angular.z = -angular
                print("右自转")
                continue

            # road banner right
            elif command == ord('l'):
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                print("停止")
                continue

            # ‘z’和‘c’用于辅助收集停止符号数据，机器走到停止符号前，左右平移，调整终止符号的位置再拍照，以获得更丰富的数据样本
            elif command == ord('z'):
                self.twist.linear.x = v
                self.twist.angular.z = angular
                continue

            elif command == ord('c'):
                self.twist.linear.x = v
                self.twist.angular.z = -angular
                continue

            elif command == ord('8'):
                num_list[0] += 1
                total_images_collected += 1
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tempdata = imgsplit.datareturn(frame)
                data.append(tempdata)
                label.append(0)


            elif command == ord('4'):
                num_list[1] += 1
                total_images_collected += 1
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tempdata = imgsplit.datareturn(frame)
                data.append(tempdata)
                label.append(1)


            elif command == ord('6'):
                num_list[2] += 1
                total_images_collected += 1
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tempdata = imgsplit.datareturn(frame)
                data.append(tempdata)
                label.append(2)

            # elif command == ord('1'):
            #     num_list[3] += 1
            #     total_images_collected += 1
            #     frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     tempdata = imgsplit.datareturn(frame)
            #     data.append(tempdata)
            #     label.append(3)
            #
            # elif command == ord('3'):
            #     num_list[3] += 1
            #     total_images_collected += 1
            #     frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     tempdata = imgsplit.datareturn(frame)
            #     data.append(tempdata)
            #     label.append(4)

            self.robot.publish_twist(self.twist)
            print('publish successful')
            self.rate_run.sleep()

        data = np.array(data)
        label = np.array(label)
        num_list = np.array(num_list)

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        try:
            np.savez(self.data_path + '/' + self.saved_file_name + '.npz', train=data, train_labels=label,
                     num_list=num_list)
            print("total collect img: " + str(total_images_collected))
            print('data collect successful')
        except IOError as e:
            print(e)


if __name__ == '__main__':
    CollectTrainingData()
