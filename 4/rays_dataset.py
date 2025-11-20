import numpy as np
import torch
from rays_geometry import pixel_to_ray
from sample_along_rays import sample_along_rays

class RaysData:
    def __init__(self, images, K, c2ws):
        """
        Args:
            images: (N_img, H, W, 3) numpy array, 已经 /255.0 归一化
            K: (3,3) 或 (N_img,3,3) 相机内参
            c2ws: (N_img,4,4) 相机到世界变换矩阵
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        self.K = torch.tensor(K, dtype=torch.float32)
        self.c2ws = torch.tensor(c2ws, dtype=torch.float32)
        self.N_img, self.H, self.W, _ = self.images.shape

        # ==== 预计算所有像素坐标 (u,v) ====
        us, vs = torch.meshgrid(
            torch.arange(self.W), torch.arange(self.H), indexing="xy"
        )
        uvs = torch.stack([us, vs], dim=-1).reshape(-1, 2)  # (H*W,2)
        self.uvs = uvs + 0.5  # 加 0.5：像素中心偏移
        self.pixels = self.images.reshape(self.N_img, -1, 3)  # (N_img, H*W, 3)

    def sample_rays(self, n_rays):
        """
        从所有视角中随机采样 n_rays 条光线。
        返回:
            rays_o: (n_rays, 3)
            rays_d: (n_rays, 3)
            pixels: (n_rays, 3)
        """
        device = self.images.device

        # 随机选取相机和像素索引
        img_ids = torch.randint(0, self.N_img, (n_rays,), device=device)
        pix_ids = torch.randint(0, self.H * self.W, (n_rays,), device=device)

        uvs = self.uvs[pix_ids].to(device)
        colors = self.pixels[img_ids, pix_ids].to(device)

        rays_o_list, rays_d_list = [], []
        for i in range(n_rays):
            K_i = self.K if self.K.ndim == 2 else self.K[img_ids[i]]
            c2w_i = self.c2ws[img_ids[i]]
            ray_o, ray_d = pixel_to_ray(K_i, c2w_i, uvs[i : i + 1])
            rays_o_list.append(ray_o)
            rays_d_list.append(ray_d)

        rays_o = torch.cat(rays_o_list, dim=0)
        rays_d = torch.cat(rays_d_list, dim=0)

        return rays_o, rays_d, colors
