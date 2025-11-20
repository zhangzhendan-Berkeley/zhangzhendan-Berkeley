import time
import numpy as np
import viser
import torch
from rays_geometry import pixel_to_ray
from sample_along_rays import sample_along_rays

# === 载入数据 ===
data = np.load("lego_200x200.npz")
images_train = data["images_train"] / 255.0
c2ws_train = data["c2ws_train"]
focal = data["focal"]

H, W = images_train.shape[1:3]
K = np.array([[focal, 0, W / 2],
              [0, focal, H / 2],
              [0, 0, 1]], dtype=np.float32)

# === 选择第0个相机 ===
idx = 0
img = images_train[idx]
c2w = c2ws_train[idx]

# === 从这张图随机采样100个像素 ===
num_rays = 100
us = np.random.randint(0, W, size=num_rays)
vs = np.random.randint(0, H, size=num_rays)
uvs = np.stack([us + 0.5, vs + 0.5], axis=1)  # 像素中心
uvs_t = torch.tensor(uvs, dtype=torch.float32)

# === 转为光线 ===
ray_o, ray_d = pixel_to_ray(
    torch.tensor(K, dtype=torch.float32),
    torch.tensor(c2w, dtype=torch.float32),
    uvs_t.float()
)


# === 沿光线采样 ===
points, t_vals = sample_along_rays(ray_o, ray_d, n_samples=16, near=2.0, far=6.0, perturb=False)

print(f"ray_o shape: {ray_o.shape}, ray_d shape: {ray_d.shape}")
print(f"points shape: {points.shape}")

# === viser 可视化 ===
server = viser.ViserServer(share=False)
print("Open http://localhost:8080")

# 相机锥体
server.scene.add_camera_frustum(
    f"/camera/{idx}",
    fov=2 * np.arctan2(H / 2, K[0, 0]),
    aspect=W / H,
    scale=0.15,
    wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
    position=c2w[:3, 3],
    image=img
)

# 光线
for i, (o, d) in enumerate(zip(ray_o, ray_d)):
    server.scene.add_spline_catmull_rom(
        f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
    )

# 采样点（转为 numpy）
points_np = points.cpu().numpy().reshape(-1, 3)
server.scene.add_point_cloud(
    "/samples",
    colors=np.zeros_like(points_np),
    points=points_np,
    point_size=0.03,
)

while True:
    time.sleep(0.1)
