import numpy as np
from skimage.feature import corner_harris, peak_local_max
import cv2
import matplotlib.pyplot as plt
from skimage import data
from harris import preprocessing
from auto_match import match_features,draw_matches
from computeH import ransac_homography
from main import warpImageBilinear,warpImageNearestNeighbor
from blend import create_alpha_mask, blend_images, compute_mosaic_canvas

if __name__ == "__main__":
    img_left = cv2.imread("sky_left.png")
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    coordsL, descL = preprocessing(img_left)
    print(descL.shape)

    img_right = cv2.imread("sky_right.png")
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    coordsR, descR = preprocessing(img_right)
    print(descR.shape)

    matches = match_features(descL, descR, threshold=0.75)
    left_pts = np.array([[x, y] for y, x in coordsL[matches[:, 0]]], dtype=np.float32)
    right_pts = np.array([[x, y] for y, x in coordsR[matches[:, 1]]], dtype=np.float32)

    H, inliers = ransac_homography(left_pts, right_pts)
    print("Estimated Homography:\n", H)
    print(inliers)
    print(matches.shape)

    # 可视化内点
    inlier_matches = matches[inliers]
    draw_matches(img_left, coordsL, img_right, coordsR, inlier_matches, num_show=50)

    canvas_shape, T = compute_mosaic_canvas(img_left, img_right, H)

    # warp 左图
    im_warp_bil = warpImageBilinear(img_left, T @ H, canvas_shape)
    # warp 右图
    im_right_shift = warpImageBilinear(img_right, T, canvas_shape)

    # alpha mask
    alpha_left = create_alpha_mask(*canvas_shape)
    alpha_right = create_alpha_mask(*canvas_shape)

    # blending
    mosaic = blend_images(im_right_shift, im_warp_bil, alpha_right, alpha_left)

    plt.imshow(mosaic)
    plt.title("Feathered Mosaic")
    plt.axis("off")
    plt.show()