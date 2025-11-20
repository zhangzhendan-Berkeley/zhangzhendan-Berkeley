import numpy as np
import cv2
from computeH import ransac_homography

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

import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ---------- Step 1: 读取 JSON ----------
    with open("sky_correspond.json", "r") as f:
        correspond = json.load(f)

    left_pts = np.array(correspond["im1Points"], dtype=np.float32)
    right_pts = np.array(correspond["im2Points"], dtype=np.float32)

    # ---------- Step 2: 计算 Homography (RANSAC) ----------
    H, mask = ransac_homography(left_pts, right_pts)
    print("Estimated Homography:\n", H)

    # ---------- Step 3: 读取图像 ----------
    im_left = cv2.imread("sky_left.png")[:, :, ::-1]   # BGR → RGB
    im_right = cv2.imread("sky_right.png")[:, :, ::-1]

    from blend import create_alpha_mask, blend_images, compute_mosaic_canvas

    canvas_shape, T = compute_mosaic_canvas(im_left, im_right, H)

    # warp 左图
    im_warp_bil = warpImageBilinear(im_left, T @ H, canvas_shape)
    # warp 右图（只需要平移）
    im_right_shift = warpImageBilinear(im_right, T, canvas_shape)

    # alpha mask
    alpha_left = create_alpha_mask(*canvas_shape)
    alpha_right = create_alpha_mask(*canvas_shape)

    # blending
    mosaic = blend_images(im_right_shift, im_warp_bil, alpha_right, alpha_left)

    plt.imshow(mosaic)
    plt.title("Feathered Mosaic")
    plt.axis("off")
    plt.show()