# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import time
import viser
import os

# ============================================================
# 0. Load Intrinsics
# ============================================================
data = np.load("camera_calibration.npz")
K = data["K"]
dist = data["dist"]
print("[INFO] Loaded intrinsics:")
print("K =\n", K)
print("dist =", dist.ravel())

os.makedirs("poses", exist_ok=True)

# ============================================================
# 1. Define ArUco Tag 3D Corner Coordinates (world coordinates)
#    (tag is 2cm x 2cm in this example)
# ============================================================
tag_size = 0.04  # meters
s = tag_size / 2
objp = np.array([
    [-s,  s, 0],   # TL
    [ s,  s, 0],   # TR
    [ s, -s, 0],   # BR
    [-s, -s, 0],   # BL
], dtype=np.float32)

# ============================================================
# 2. Initialize ArUco Detector
# ============================================================
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# ============================================================
# 3. Estimate Camera Pose for Each Image
# ============================================================
images = sorted(glob.glob("scan_downsampled/*.jpg"))
print(f"[INFO] Found {len(images)} images")

camera_poses = []   # list of (c2w, undistorted_image)

for i, path in enumerate(images):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Cannot load {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Detect ArUco ---
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is None or len(ids) == 0:
        print(f"[WARN] No tag found in {path}, skipped.")
        continue

    target_id = 0  # 你打印的标签 ID
    flat_ids = ids.flatten()

    if target_id not in flat_ids:
        print(f"[WARN] Tag {target_id} not found in {path}, skipped.")
        continue

    idx = np.where(flat_ids == target_id)[0][0]
    imgp = corners[idx].reshape(-1, 2).astype(np.float32)

    # --- Solve PnP (world→camera) ---
    ret = cv2.solvePnPGeneric(
        objp, imgp, K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    # ret 的长度可能是 3, 4, 或 6
    # 最前面永远是：
    #   ret[0] = success
    #   ret[1] = rvecs（list）
    #   ret[2] = tvecs（list）

    success = ret[0]
    rvecs = ret[1]
    tvecs = ret[2]

    if not success:
        print("IPPE solvePnPGeneric failed")
        continue

    # 拿两个解：
    # rvecs[0], tvecs[0] = 主解
    # rvecs[1], tvecs[1] = 镜像解（法向反向）

    R1, _ = cv2.Rodrigues(rvecs[0])
    t1 = tvecs[0].reshape(3)

    R2, _ = cv2.Rodrigues(rvecs[1])
    t2 = tvecs[1].reshape(3)

    # 计算两个解的相机位置（OpenCV w2c → c2w）
    cam1 = -R1.T @ t1
    cam2 = -R2.T @ t2

    # 选择 “相机在 tag 上方（z > 0）” 的解
    if cam1[2] > cam2[2]:
        R, t = R1, t1
    else:
        R, t = R2, t2

    # 构造 c2w
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t

    camera_poses.append((c2w, img))
    np.savez(f"poses/pose_{i:03d}.npz", c2w=c2w, image_path=path)

    print(f"[INFO] Pose {i:03d} OK, t = {t.ravel()}")

print(f"[INFO] {len(camera_poses)} valid poses.")

# ============================================================
# 4. Visualize with Viser
# ============================================================
server = viser.ViserServer(share=False)
print("[INFO] Launching Viser...")

for i, (c2w, img) in enumerate(camera_poses):
    H, W = img.shape[:2]
    fov = 2 * np.arctan2(H / 2, K[1, 1])
    aspect = W / H

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    server.scene.add_camera_frustum(
        f"/cameras/{i}",
        fov=float(fov),
        aspect=float(aspect),
        scale=0.02,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
        image=img_rgb
    )

print("[INFO] All frustums added. Explore in browser.")
while True:
    time.sleep(0.1)
