import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

gaussian_kernel_1d = cv2.getGaussianKernel(ksize=5, sigma=1)
gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T

Dx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
Dy = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])

DoGx = convolve2d(gaussian_kernel_2d, Dx, mode="full")
DoGy = convolve2d(gaussian_kernel_2d, Dy, mode="full")

for i, (kernel, name) in enumerate([(DoGx, "DoGx"), (DoGy, "DoGy")]):

    kernel_img = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX)
    kernel_img = kernel_img.astype(np.uint8)

    cv2.imwrite(f"{name}.jpg", kernel_img)

    plt.subplot(1, 2, i + 1)
    plt.imshow(kernel_img, cmap="gray")
    plt.title(name)
    plt.axis("off")

plt.show()
