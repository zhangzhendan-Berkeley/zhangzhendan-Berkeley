import time
import numpy as np
import viser
import torch
from rays_dataset import RaysData
from sample_along_rays import sample_along_rays

# === 载入 lego 数据集 ===
data = np.load("dataset/my_data.npz")
# data = np.load("lego_200x200.npz")
images_train = data["images_train"] / 255.0
c2ws_train = data["c2ws_train"]
focal = data["focal"]

H, W = images_train.shape[1:3]
# K = np.array([[focal, 0, W / 2],
#               [0, focal, H / 2],
#               [0, 0, 1]], dtype=np.float32)

K = data["K"]
dist = data["dist"]
print("[INFO] Loaded intrinsics:")
print("K =\n", K)
print("dist =", dist.ravel())

dataset = RaysData(images_train, K, c2ws_train)

# === 采样光线 ===
rays_o, rays_d, pixels = dataset.sample_rays(100)
points, _ = sample_along_rays(rays_o, rays_d, perturb=False, near=0.001, far=0.5)

# === 可视化 ===
server = viser.ViserServer(share=False)
print("Open in browser: http://localhost:8080 (or the port viser prints)")



# 添加相机锥体
for i, (img, c2w) in enumerate(zip(images_train, c2ws_train)):
    img_rgb = img[..., ::-1]  # swap channels
    server.scene.add_camera_frustum(
        f"/cameras/{i}",
        fov=2 * np.arctan2(H / 2, K[0, 0]),
        aspect=W / H,
        scale=0.02,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
        image=img_rgb,
    )

# 添加光线
for i, (o, d) in enumerate(zip(rays_o, rays_d)):
    server.scene.add_spline_catmull_rom(
        f"/rays/{i}", positions=np.stack((o, o + d * 4.0)),
    )

points_np = points.cpu().numpy().reshape(-1, 3)
server.scene.add_point_cloud(
    "/samples",
    colors=np.zeros_like(points_np),
    points=points_np,
    point_size=0.005,
)

while True:
    time.sleep(0.1)
