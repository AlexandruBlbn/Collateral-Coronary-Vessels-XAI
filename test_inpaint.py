import cv2
import numpy as np

img = cv2.imread('data/extra/trainB/0.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))

_, fg = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

# Fill black borders
img_padded = cv2.inpaint(img, 255-fg, 5, cv2.INPAINT_TELEA)

cv2.imwrite("data/test_inpaint.png", img_padded)
