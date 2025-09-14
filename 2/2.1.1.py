import cv2
import numpy as np
from scipy.signal import convolve2d

img = cv2.imread("taj.jpg", cv2.IMREAD_GRAYSCALE).astype(float)

gaussian_kernel_1d = cv2.getGaussianKernel(ksize=9, sigma=2)
G = gaussian_kernel_1d @ gaussian_kernel_1d.T
G /= G.sum()

I_blur = convolve2d(img, G, mode="same", boundary="symm")
H = img - I_blur

alpha = 1.5
I_sharp = img + alpha * H

I_sharp = np.clip(I_sharp, 0, 255).astype(np.uint8)

cv2.imwrite("taj_sharp.jpg", I_sharp)
