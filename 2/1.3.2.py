import cv2
import numpy as np
from scipy.signal import convolve2d

gaussian_kernel_1d = cv2.getGaussianKernel(ksize=5, sigma=1)
gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

Dx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
Dy = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]])

Kx = convolve2d(gaussian_kernel_2d, Dx, mode="full")
Ky = convolve2d(gaussian_kernel_2d, Dy, mode="full")

img = cv2.imread("myself.jpg", cv2.IMREAD_GRAYSCALE)

Ix = convolve2d(img, Kx, mode="same", boundary="fill", fillvalue=0)
Iy = convolve2d(img, Ky, mode="same", boundary="fill", fillvalue=0)

grad = np.sqrt(Ix**2 + Iy**2)

grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

thresh_value = 30
_, binary = cv2.threshold(grad, thresh_value, 255, cv2.THRESH_BINARY)

cv2.imwrite("myself_Dog.jpg", binary)