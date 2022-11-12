import math
import cv2
import numpy as np

x = [1,-1 ]
center = [0, 0]
r = math.sqrt(math.pow(x[0] - center[0], 2) + math.pow(x[1] - center[1], 2))
theta = math.atan2(x[1] - center[1], x[0] - center[0]) / math.pi
if theta>=0:
    theta=theta
else:
    theta=2+theta
print(r, theta)
