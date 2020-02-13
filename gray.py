import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('images/class-4/OK/76.jpg')

# get image's size
height, width = img.shape[:2]
# split channels
blue, green, red = cv2.split(img)

# convert rgb to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV
# split hsv image's channels
h, s, v = cv2.split(hsv)

# for zero padding
int_zeros = np.zeros((height, width), 'uint8')
# merge to (0, 0, v), and hsv is converted rgb
cv2.imshow("hoge", cv2.cvtColor(cv2.merge([int_zeros, int_zeros, v]), cv2.COLOR_HSV2BGR))
cv2.waitKey(0)
