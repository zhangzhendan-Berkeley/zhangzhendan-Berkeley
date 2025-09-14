import numpy as np
import math
from PIL.DdsImagePlugin import DXGI_FORMAT

def conv2d(image, kernel):
    H, W = image.shape
    kH,kW = kernel.shape
    padH, padW = kH // 2, kW // 2
    kernel_flipped = np.flipud(np.fliplr(kernel))
    padded_image = np.pad(image, ((padH, padW), (padH, padW)), mode='constant', constant_values=0)
    output = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            sum = 0
            for m in range(kH):
                for n in range(kW):
                    sum += padded_image[i+m, j+n] * kernel_flipped[m, n]
            output[i, j] = sum

    return output

def conv2d_fast(image, kernel):
    H, W = image.shape
    kH,kW = kernel.shape
    padH, padW = kH // 2, kW // 2
    kernel_flipped = np.flipud(np.fliplr(kernel))
    padded_image = np.pad(image, ((padH, padW), (padH, padW)), mode='constant', constant_values=0)
    output = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            region = padded_image[i : i + kH, j : j + kW]
            output[i, j] = np.sum(region * kernel_flipped)

    return output

image = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])   # Sobel-like operator

print(conv2d_fast(image,kernel))

import cv2

img = cv2.imread("myself.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("myself_gray.jpg", gray)

img = cv2.imread("myself.jpg", cv2.IMREAD_GRAYSCALE)

myKernel = np.outer(np.arange(-4,5), np.ones(9))
MyImage = conv2d_fast(img, myKernel)
MyImage = cv2.normalize(MyImage, None, 0, 255, cv2.NORM_MINMAX)
MyImage = MyImage.astype(np.uint8)
cv2.imwrite("myself_conved.jpg", MyImage)

Dx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])   # Sobel-like operator
Dy = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])

DxImage = conv2d_fast(img, Dx)
DxImage_show = cv2.normalize(DxImage, None, 0, 255, cv2.NORM_MINMAX)
DxImage_show = DxImage_show.astype(np.uint8)
cv2.imwrite("myself_Dx.jpg", DxImage_show)

DyImage = conv2d_fast(img, Dy)
DyImage_show = cv2.normalize(DyImage, None, 0, 255, cv2.NORM_MINMAX)
DyImage_show = DyImage_show.astype(np.uint8)
cv2.imwrite("myself_Dy.jpg", DyImage_show)

grad = np.sqrt(DxImage.astype(float)**2 + DyImage.astype(float)**2)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
grad = grad.astype(np.uint8)
cv2.imwrite("myself_grad.jpg", grad)

