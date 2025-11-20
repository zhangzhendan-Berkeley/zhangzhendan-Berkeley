import cv2
import numpy as np
from scipy.signal import convolve2d

gaussian_kernel_1d = cv2.getGaussianKernel(ksize=5, sigma=1)
gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

img = cv2.imread("myself.jpg", cv2.IMREAD_GRAYSCALE)

blur = convolve2d(img,gaussian_kernel_2d, mode="same", boundary="fill", fillvalue=0)

Dx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])

Dy = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]])

DxImage = convolve2d(blur, Dx, mode="same", boundary="fill", fillvalue=0)
DyImage = convolve2d(blur, Dy, mode="same", boundary="fill", fillvalue=0)

grad = np.sqrt(DxImage.astype(float)**2 + DyImage.astype(float)**2)

grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
grad = grad.astype(np.uint8)

thresh_value = 20
_, binary = cv2.threshold(grad, thresh_value, 255, cv2.THRESH_BINARY)

cv2.imwrite("myself_blur_grad_binary.jpg", binary)

