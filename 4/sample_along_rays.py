# sample_along_rays.py
import torch

def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    perturb: bool = True,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Uniform or importance sampling of 3D points along rays.
    Args:
        rays_o: (B, 3) 光线原点
        rays_d: (B, 3) 光线方向（单位向量）
        n_samples: 每条光线上的采样点数
        near, far: 采样深度范围
        perturb: 是否添加随机扰动（用于训练）
        weights: (B, M) 可选，若提供则基于 coarse 权重做重要性重采样

    Returns:
        points: (B, n_samples, 3) 采样的3D点
        t_vals: (B, n_samples) 深度参数
    """
    device = rays_o.device
    B = rays_o.shape[0]

    # ====== 若无权重，使用均匀采样 ======
    if weights is None:
        t_vals = torch.linspace(near, far, n_samples, device=device)
        t_vals = t_vals.expand(B, n_samples)

        if perturb:
            mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
            upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
            lower = torch.cat([t_vals[:, :1], mids], dim=-1)
            t_rand = torch.rand(t_vals.shape, device=device)
            t_vals = lower + (upper - lower) * t_rand
    else:
        # ====== 若有权重，按 PDF / CDF 重采样 ======
        weights = weights + 1e-8  # 防止除0
        pdf = weights / torch.sum(weights, dim=1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # (B, M+1)

        # 在 [0,1] 均匀采样 n_samples 个随机数
        u = torch.rand(B, n_samples, device=device)
        inds = torch.searchsorted(cdf, u, right=True)
        inds = torch.clamp(inds, 1, cdf.shape[-1]-1)

        below = inds - 1
        above = inds
        cdf_g = torch.gather(cdf, 1, above) - torch.gather(cdf, 1, below)
        denom = torch.where(cdf_g < 1e-5, torch.ones_like(cdf_g), cdf_g)
        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, above)
        u_expand = u
        t = (u_expand - cdf_below) / denom
        # 将 [0,1] 映射回 [near, far]
        bins = torch.linspace(near, far, weights.shape[1], device=device)
        bins_below = torch.gather(bins.expand(B, -1), 1, below)
        bins_above = torch.gather(bins.expand(B, -1), 1, above)
        t_vals = bins_below + t * (bins_above - bins_below)

    # ===== 计算采样点 =====
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)
    return points, t_vals

