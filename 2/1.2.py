import cv2
import numpy as np
from scipy.signal import convolve2d

img = cv2.imread("myself.jpg", cv2.IMREAD_GRAYSCALE)

Dx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])

Dy = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]])

DxImage = convolve2d(img, Dx, mode="same", boundary="fill", fillvalue=0)
DyImage = convolve2d(img, Dy, mode="same", boundary="fill", fillvalue=0)

grad = np.sqrt(DxImage.astype(float)**2 + DyImage.astype(float)**2)

grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
grad = grad.astype(np.uint8)

thresh_value = 20

_, binary = cv2.threshold(grad, thresh_value, 255, cv2.THRESH_BINARY)

cv2.imwrite("myself_grad_binary.jpg", binary)