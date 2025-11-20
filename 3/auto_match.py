import numpy as np
from skimage.feature import corner_harris, peak_local_max
import cv2
import matplotlib.pyplot as plt
from skimage import data
from harris import preprocessing

def match_features(desc_left, desc_right, threshold=0.8):
    """
    Match descriptors using Lowe's ratio test
    """
    matches = []

    for i, d_left in enumerate(desc_left):
        # 计算欧氏距离到右图所有 descriptor
        dists = np.linalg.norm(desc_right - d_left, axis=1)

        # 找最小和次小距离
        if len(dists) < 2:
            continue
        sorted_idx = np.argsort(dists)
        d1, d2 = dists[sorted_idx[0]], dists[sorted_idx[1]]

        # ratio test
        if d1 / (d2 + 1e-10) < threshold:
            matches.append((i, sorted_idx[0]))

    return np.array(matches)

def draw_matches(img1, coords1, img2, coords2, matches, num_show=50):
    """Draw matched features"""
    num_show = min(num_show, len(matches))
    selected = matches[np.random.choice(len(matches), num_show, replace=False)]

    # 拼接左右图
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2), dtype=img1.dtype)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas, cmap='gray')

    for i, j in selected:
        y1, x1 = coords1[i]
        y2, x2 = coords2[j]
        plt.plot([x1, x2 + w1], [y1, y2], 'r', linewidth=0.5)
        plt.scatter([x1, x2 + w1], [y1, y2], c='lime', s=5)

    plt.axis('off')
    plt.title(f"{num_show} matched features")
    plt.show()

if __name__ == "__main__":
    img_left = cv2.imread("room_left.png")
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    coordsL, descL = preprocessing(img_left)
    print(descL.shape)

    img_right = cv2.imread("room_right.png")
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    coordsR, descR = preprocessing(img_right)
    print(descR.shape)

    matches = match_features(descL, descR, threshold=0.75)
    draw_matches(img_left, coordsL, img_right, coordsR, matches)