import numpy as np
import cv2
from computeH import ransac_homography
import json
import matplotlib.pyplot as plt

def warpImageNearestNeighbor(im, H, out_shape=None):
    h, w = im.shape[:2]
    if out_shape is None:
        out_shape = (h, w)
    out_h, out_w = out_shape
    H_inv = np.linalg.inv(H)

    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for y in range(out_h):
        for x in range(out_w):
            src = H_inv @ np.array([x, y, 1])
            src /= src[2]
            sx, sy = int(round(src[0])), int(round(src[1]))
            if 0 <= sx < w and 0 <= sy < h:
                warped[y, x] = im[sy, sx]
    return warped

def warpImageBilinear(im, H, out_shape=None):
    h, w = im.shape[:2]
    if out_shape is None:
        out_shape = (h, w)
    out_h, out_w = out_shape
    H_inv = np.linalg.inv(H)

    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for y in range(out_h):
        for x in range(out_w):
            src = H_inv @ np.array([x, y, 1])
            src /= src[2]
            sx, sy = src[0], src[1]

            if 0 <= sx < w-1 and 0 <= sy < h-1:
                x0, y0 = int(np.floor(sx)), int(np.floor(sy))
                dx, dy = sx - x0, sy - y0

                top = (1-dx)*im[y0, x0] + dx*im[y0, x0+1]
                bottom = (1-dx)*im[y0+1, x0] + dx*im[y0+1, x0+1]
                warped[y, x] = (1-dy)*top + dy*bottom
    return warped


# ---------- Step 1: 读取 JSON ----------
with open("street_correspond.json", "r") as f:
    correspond = json.load(f)

left_pts = np.array(correspond["im1Points"], dtype=np.float32)
right_pts = np.array(correspond["im2Points"], dtype=np.float32)

# ---------- Step 2: 计算 Homography (RANSAC) ----------
H, mask = ransac_homography(left_pts, right_pts)
print("Estimated Homography:\n", H)

# ---------- Step 3: 读取图像 ----------
im_left = cv2.imread("street_left.png")[:,:,::-1]   # BGR → RGB
im_right = cv2.imread("street_right.png")[:,:,::-1]

from blend import compute_mosaic_canvas

# ---------- Step 4: 确定输出画布 ----------
canvas_shape, T = compute_mosaic_canvas(im_left, im_right, H)

# ---------- Step 5: 分别做 warping ----------
im_warp_nn = warpImageNearestNeighbor(im_left, T @ H, canvas_shape)
im_warp_bil = warpImageBilinear(im_left, T @ H, canvas_shape)

# ---------- Step 6: 可视化 & 保存 ----------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(im_warp_nn)
plt.title("Warped (Nearest Neighbor)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(im_warp_bil)
plt.title("Warped (Bilinear)")
plt.axis("off")

plt.tight_layout()
plt.show()

# # 保存结果，后面做 blending 用
# cv2.imwrite("warp_nn.png", im_warp_nn[:,:,::-1])   # RGB → BGR
# cv2.imwrite("warp_bilinear.png", im_warp_bil[:,:,::-1])
# print("Saved warped images: warp_nn.png, warp_bilinear.png")
