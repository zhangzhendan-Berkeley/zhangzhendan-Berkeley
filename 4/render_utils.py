# render_utils.py
import torch
import numpy as np
from typing import Optional
from rays_geometry import pixel_to_ray
from sample_along_rays import sample_along_rays
from volume_render import volrend

@torch.no_grad()
def render_one_view(
    model,
    K: np.ndarray,
    c2w: np.ndarray,
    H: int,
    W: int,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    chunk: int = 8192,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """
    Render a single image from a given camera by marching rays and volume rendering.

    Returns:
        image: (H, W, 3) float32 in [0,1]
    """
    model.eval()
    # 1) Build per-pixel UV grid (pixel centers)
    us = torch.arange(W, dtype=torch.float32) + 0.5
    vs = torch.arange(H, dtype=torch.float32) + 0.5
    grid_u, grid_v = torch.meshgrid(us, vs, indexing="xy")
    uvs = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1)  # (H*W, 2)

    # 2) Rays (world)
    K_t   = torch.tensor(K, dtype=torch.float32, device=device)
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    uvs_t = uvs.to(device)
    rays_o, rays_d = pixel_to_ray(K_t, c2w_t, uvs_t)  # (N,3),(N,3)
    N = rays_o.shape[0]

    # 3) March and render in chunks
    step_size = (far - near) / n_samples
    out = []
    for i in range(0, N, chunk):
        o = rays_o[i:i+chunk]
        d = rays_d[i:i+chunk]

        # Points along rays
        pts, _ = sample_along_rays(o, d, n_samples=n_samples, near=near, far=far, perturb=False)  # (B,S,3)
        # Network: sigma & rgb
        sigma, rgb = model(pts, d.unsqueeze(1).expand_as(pts))  # (B,S,1),(B,S,3)

        # Volume render
        colors = volrend(sigma, rgb, step_size=step_size)  # (B,3)
        out.append(colors)

    img = torch.cat(out, dim=0).reshape(H, W, 3).clamp(0, 1)
    return img.detach().cpu().numpy().astype(np.float32)
