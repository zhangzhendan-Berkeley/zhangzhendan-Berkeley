import cv2
import numpy as np
from scipy.signal import convolve2d

img = cv2.imread("myself.jpg", cv2.IMREAD_GRAYSCALE).astype(float)

gaussian_kernel_1d = cv2.getGaussianKernel(ksize=9, sigma=2)
G = gaussian_kernel_1d @ gaussian_kernel_1d.T
G /= G.sum()

size = 9
delta = np.zeros((size, size), dtype=np.float32)
delta[size // 2, size // 2] = 1.0
alpha = 1.5  # 可调
X = (1 + alpha) * delta - alpha * G

I_sharp = convolve2d(img, X, mode="same", boundary="symm")

I_sharp = np.clip(I_sharp, 0, 255).astype(np.uint8)

cv2.imwrite("myself_sharp_onestep.jpg", I_sharp)
