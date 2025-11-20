# rays_geometry.py
import torch
from typing import Tuple

def _to_homo(x: torch.Tensor) -> torch.Tensor:
    """[... , 3] -> [..., 4]  末尾拼 1"""
    ones = torch.ones_like(x[..., :1])
    return torch.cat([x, ones], dim=-1)

def _from_homo(xh: torch.Tensor) -> torch.Tensor:
    """[... , 4] -> [..., 3]  除以最后一维"""
    w = xh[..., -1:]
    return xh[..., :3] / w

def _normalize(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)

# -------------------------------
# 1) Camera -> World transform
# -------------------------------
def transform_points(c2w: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
    """
    将相机坐标系中的点变到世界坐标系。
    c2w: (..., 4, 4)  相机到世界的齐次矩阵
    x_c: (..., N, 3) 相机坐标的点
    return: (..., N, 3) 世界坐标的点
    """
    xh = _to_homo(x_c)                                  # (..., N, 4)
    # 右乘：[..., N, 4] @ [..., 4, 4]^T 以便 broadcast
    xw_h = torch.matmul(xh, c2w.transpose(-1, -2))      # (..., N, 4)
    return _from_homo(xw_h)                              # (..., N, 3)

def transform_points_inv(c2w: torch.Tensor, x_w: torch.Tensor) -> torch.Tensor:
    """
    世界 -> 相机（使用 c2w 的逆矩阵）
    """
    w2c = torch.linalg.inv(c2w)
    xh = _to_homo(x_w)
    xc_h = torch.matmul(xh, w2c.transpose(-1, -2))
    return _from_homo(xc_h)

# ----------------------------------------
# 2) Pixel -> Camera (pinhole backproject)
# ----------------------------------------
def pixel_to_camera(
    K: torch.Tensor, uv: torch.Tensor, depth: torch.Tensor
) -> torch.Tensor:
    """
    将像素坐标反投影到相机坐标系下的 3D 点。
    K: (..., 3, 3) 内参；支持单个或批量
    uv: (..., N, 2) 像素坐标 (u, v) —— 记得外面构造时加 0.5 做像素中心偏移
    depth: (..., N) 或标量  深度 z (>0)
    return: (..., N, 3) 相机坐标系下 3D 点
    公式：x_c = z * K^{-1} [u, v, 1]^T
    """
    # 保证 batch 维度能 broadcast
    if depth.ndim == uv.ndim - 1:
        depth = depth.unsqueeze(-1)  # (..., N, 1)

    # 构造齐次像素
    ones = torch.ones_like(uv[..., :1])
    uv1 = torch.cat([uv, ones], dim=-1)                 # (..., N, 3)

    # K^{-1}
    Kinv = torch.linalg.inv(K)                          # (..., 3, 3)
    # 右乘：[..., N, 3] @ [..., 3, 3]^T
    dirs = torch.matmul(uv1, Kinv.transpose(-1, -2))    # (..., N, 3)
    x_c = dirs * depth                                  # 缩放到给定深度
    return x_c

# ----------------------------------------
# 3) Pixel -> Ray (origin, normalized dir)
# ----------------------------------------
def pixel_to_ray(
    K: torch.Tensor, c2w: torch.Tensor, uv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将像素坐标转换为世界坐标系下的光线 (ray_o, ray_d)。
    K:   (..., 3, 3)
    c2w: (..., 4, 4)
    uv:  (..., N, 2) 像素坐标 (u, v) —— 外面采样时请加 0.5 做像素中心
    return:
      ray_o: (..., N, 3) 光线原点（相机中心）
      ray_d: (..., N, 3) 归一化光线方向（世界坐标系）
    步骤：
      1) 在相机系选取 z=1 的点：x_c = K^{-1}[u, v, 1]^T
      2) 变换到世界系：x_w = c2w * x_c
      3) 相机中心 o = c2w[:3, 3]
      4) 方向 d = normalize(x_w - o)
    """
    # 相机中心（世界系）
    o_w = c2w[..., :3, 3]                               # (..., 3)
    # 先在相机系取 z=1 的点
    ones = torch.ones_like(uv[..., :1])
    uv1 = torch.cat([uv, ones], dim=-1)                 # (..., N, 3)
    Kinv = torch.linalg.inv(K)                          # (..., 3, 3)
    xc = torch.matmul(uv1, Kinv.transpose(-1, -2))      # (..., N, 3), depth=1

    # 相机 -> 世界
    xw = transform_points(c2w, xc)                      # (..., N, 3)

    # 原点 broadcast 到 N
    ray_o = o_w.unsqueeze(-2).expand_as(xw)             # (..., N, 3)
    ray_d = _normalize(xw - ray_o)                      # (..., N, 3)
    return ray_o, ray_d


