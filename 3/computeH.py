import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# ---------- 使用RANSAC获取inliers ----------
def compute_homography(pts1, pts2):
    A = []
    for (x,y), (xp,yp) in zip(pts1, pts2):
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    U,S,Vt = np.linalg.svd(A)
    H = Vt[-1,:].reshape(3,3)
    return H/H[2,2]

def ransac_homography(pts1, pts2, num_iter=2000, thresh=5.0):
    best_inliers = []
    best_H = None
    n = pts1.shape[0]
    for _ in range(num_iter):
        idx = np.random.choice(n, 4, replace=False)
        H = compute_homography(pts1[idx], pts2[idx])
        pts1_h = np.hstack([pts1, np.ones((n,1))])
        proj = (H @ pts1_h.T).T
        proj /= proj[:,2][:,np.newaxis]
        errors = np.linalg.norm(proj[:,:2] - pts2, axis=1)
        inliers = np.where(errors < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
    return best_H, best_inliers

if __name__ == "__main__":
    # ---------- 读图和点 ----------
    left_img = cv2.imread("room_left.png")
    right_img = cv2.imread("room_right.png")

    with open("sky_correspond.json", "r") as f:
        data = json.load(f)

    pts1 = np.array(data["im1Points"], dtype=np.float32)  # left
    pts2 = np.array(data["im2Points"], dtype=np.float32)  # right

    H, inliers = ransac_homography(pts1, pts2, num_iter=2000, thresh=5.0)
    inliers = set(inliers)
    # np.set_printoptions(precision=3, suppress=True)  # 保留3位小数，并去掉科学计数法
    print(H)

    # ---------- 拼接两张图像，方便显示 ----------
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]
    canvas = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = left_img
    canvas[:h2, w1:w1+w2] = right_img

    # ---------- 画点 ----------
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        x1,y1 = int(p1[0]), int(p1[1])
        x2,y2 = int(p2[0]+w1), int(p2[1])  # 右图坐标偏移
        color = (0,255,0) if i in inliers else (0,0,255)  # green=内点, red=外点
        cv2.circle(canvas, (x1,y1), 6, color, -1)
        cv2.circle(canvas, (x2,y2), 6, color, -1)
        cv2.line(canvas, (x1,y1), (x2,y2), color, 1)

    # ---------- 显示 ----------
    plt.figure(figsize=(15,10))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Correspondences: Green=Inliers, Red=Outliers")
    plt.show()
