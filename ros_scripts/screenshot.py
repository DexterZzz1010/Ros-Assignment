import cv2
import numpy as np

import pyscreenshot as ImageGrab

# image1 = pyautogui.screenshot(region=[1000, 20, 960, 1000])
size = (960, 20, 1800, 1000)
image1 = ImageGrab.grab(size)
image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
cv2.imshow("Screenshot", image1)
cv2.waitKey()
cv2.destroyAllWindows()
