import cv2
import numpy as np
import glob
import os
import time

# ============================================================
# Configuration
# ============================================================
IMAGE_PATTERN = "calibration_images_ds/*.jpg"   # 标定图像路径，可改为你的路径
TAG_SIZE_M = 0.054                               # 单个 ArUco 边长（米）
SAVE_PATH = "camera_calibration.npz"            # 输出文件
VISUALIZE = False                               # 是否显示检测结果

# ============================================================
# 1. 创建 ArUco 检测器
# ============================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = (cv2.aruco.DetectorParameters_create()
                if hasattr(cv2.aruco, "DetectorParameters_create")
                else cv2.aruco.DetectorParameters())

# ============================================================
# 2. 加载图片
# ============================================================
images = sorted(glob.glob(IMAGE_PATTERN))
if len(images) == 0:
    raise RuntimeError("No calibration images found!")

print(f"[INFO] Found {len(images)} images for calibration")

# ============================================================
# 3. 构建3D世界坐标（每个tag的4个角）
# ============================================================
objp = np.array([
    [0, 0, 0],
    [TAG_SIZE_M, 0, 0],
    [TAG_SIZE_M, TAG_SIZE_M, 0],
    [0, TAG_SIZE_M, 0]
], dtype=np.float32)

objpoints = []   # 存放3D点
imgpoints = []   # 存放2D点
valid_count = 0

# ============================================================
# 4. 遍历所有图片并检测角点（仅使用 ID==0 的 tag）
# ============================================================
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"[WARN] Cannot read {fname}, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        ids = ids.flatten()
        if 0 in ids:
            # 找到 id==0 的索引
            i0 = np.where(ids == 0)[0][0]
            objpoints.append(objp)
            imgpoints.append(corners[i0][0])
            valid_count += 1

            if VISUALIZE:
                cv2.aruco.drawDetectedMarkers(img, [corners[i0]], np.array([[0]]))
                cv2.imshow("Detected Tag 0", img)
                cv2.waitKey(100)
        else:
            print(f"[WARN] No tag ID=0 in image {idx+1}")
    else:
        print(f"[WARN] No tags detected in image {idx+1}")

cv2.destroyAllWindows()
print(f"[INFO] Using {valid_count} valid frames (tag ID=0 detected)")

# ============================================================
# 5. 相机标定计算
# ============================================================
if len(objpoints) < 3:
    raise RuntimeError("Not enough valid detections for calibration (need >=3 frames with tag ID=0).")

flags = (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
print(f"[INFO] Running calibration with {len(objpoints)} valid detections...")

start = time.time()
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)
print(f"[INFO] Calibration finished in {time.time()-start:.2f}s")

# ============================================================
# 6. 保存与输出结果
# ============================================================
np.savez(SAVE_PATH, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, error=ret)
print(f"[SUCCESS] Calibration done! Saved to {SAVE_PATH}")
print("[RESULT] Reprojection error:", ret)
print("[INTRINSIC MATRIX K]:\n", K)
print("[DISTORTION COEFFS]:", dist.ravel())
