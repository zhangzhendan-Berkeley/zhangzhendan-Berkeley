# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import os

# ============================================================
# 0. 加载相机内参（来自 Part 0.1）
# ============================================================
CALIB_PATH = "camera_calibrationa.npz"
POSE_DIR   = "poses"
OUT_DIR    = "dataset"
OUT_FILE   = os.path.join(OUT_DIR, "my_data.npz")

calib = np.load(CALIB_PATH)
K_orig = calib["K"].astype(np.float32)
dist   = calib["dist"].astype(np.float32)

print("[INFO] Loaded intrinsics K:")
print(K_orig)
print("dist =", dist.ravel())

# ============================================================
# 1. 加载姿态（poses/*.npz）
# ============================================================
pose_data = []
for npz_file in sorted(glob.glob(os.path.join(POSE_DIR, "*.npz"))):
    d = np.load(npz_file, allow_pickle=True)
    c2w = d["c2w"].astype(np.float32)

    if "image_path" not in d:
        print(f"[WARN] {npz_file} missing image_path")
        continue

    path = d["image_path"].item()
    if not os.path.exists(path):
        print(f"[WARN] Image {path} missing")
        continue

    pose_data.append((c2w, path))

if len(pose_data) == 0:
    raise RuntimeError("[ERROR] No pose files!")

print(f"[INFO] Found {len(pose_data)} pose files.")

# ============================================================
# 2. 去畸变 + 统一尺寸 + 同步更新 K
# ============================================================
images_out = []
c2ws_out   = []

target_size = None     # (W, H)
K_final     = None     # undistort + resize 后的内参

for i, (c2w, path) in enumerate(pose_data):
    img = cv2.imread(path)
    if img is None:
        print("[WARN] failed to read", path)
        continue

    H0, W0 = img.shape[:2]

    # (1) undistort 必须使用新 K
    K_new, _ = cv2.getOptimalNewCameraMatrix(
        K_orig, dist, (W0, H0), alpha=0
    )
    undist = cv2.undistort(img, K_orig, dist, None, K_new)

    # (2) 统一尺寸
    if target_size is None:
        target_size = (undist.shape[1], undist.shape[0])   # (W,H)
        K_final = K_new.copy()
        print(f"[INFO] Target size = {target_size}")
    else:
        if (undist.shape[1], undist.shape[0]) != target_size:
            # resize + scale K_new
            scale_w = target_size[0] / undist.shape[1]
            scale_h = target_size[1] / undist.shape[0]

            undist = cv2.resize(undist, target_size, interpolation=cv2.INTER_AREA)

            K_new[0,0] *= scale_w
            K_new[1,1] *= scale_h
            K_new[0,2] *= scale_w
            K_new[1,2] *= scale_h

        # 第一张的 K_new 是最终 K_final，其余保持一致即可
        K_final = K_final  # 不变

    # (3) BGR → RGB，[0,255]
    # rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    rgb = undist
    rgb = rgb.astype(np.float32)

    images_out.append(rgb)
    c2ws_out.append(c2w)

images_out = np.array(images_out)       # (N,H,W,3)
c2ws_out   = np.array(c2ws_out)         # (N,4,4)
N, H, W, _ = images_out.shape

print(f"[INFO] Undistorted {N} images, final resolution = {H}×{W}")
print("[INFO] Final K:")
print(K_final)

# ============================================================
# 3. 划分 train / val / test（test = 所有图像）
# ============================================================
idxs = np.arange(N)
np.random.shuffle(idxs)

N_train = int(1.0 * N)
N_val   = int(0.0 * N)
N_test  = N      # ✅ test 全部图像

train_idx = idxs[:N_train]
val_idx   = idxs[N_train:N_train+N_val]
test_idx  = np.arange(N)  # 顺序不变（你原来要求的）

images_train = images_out[train_idx]
c2ws_train   = c2ws_out[train_idx]

images_val   = images_out[val_idx]
c2ws_val     = c2ws_out[val_idx]

images_test  = images_out[test_idx]
c2ws_test    = c2ws_out[test_idx]

print(f"[INFO] Split: train={len(train_idx)}, val={len(val_idx)}, test(all)={N_test}")

# ============================================================
# 4. 计算 focal（与你原来一致）
# ============================================================
focal = float((K_final[0,0] + K_final[1,1]) / 2.0)
print(f"[INFO] focal = {focal:.3f}")

# ============================================================
# 5. 保存 npz（字段完全兼容你旧代码）
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)

np.savez(
    OUT_FILE,

    images_train=images_train.astype(np.float32),
    c2ws_train=c2ws_train.astype(np.float32),

    images_val=images_val.astype(np.float32),
    c2ws_val=c2ws_val.astype(np.float32),

    images_test=images_test.astype(np.float32),
    c2ws_test=c2ws_test.astype(np.float32),

    focal=np.float32(focal),          # ✅ 保留你原来的字段
    K=K_final.astype(np.float32),     # ✅ undistort + resize 后的真实内参
    dist=np.zeros_like(dist),         # ✅ undistort 后畸变全为 0
    H=np.int32(H),
    W=np.int32(W),
)

print(f"[INFO] Saved dataset → {OUT_FILE}")
print("[DONE] Part 0.4 dataset creation complete!")


data = np.load("dataset/my_data.npz")
print("Loaded K =\n", data["K"])