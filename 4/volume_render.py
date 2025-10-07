# volume_render.py
import torch

def volrend(sigmas: torch.Tensor, rgbs: torch.Tensor, step_size: float) -> torch.Tensor:
    """
    Volumetric rendering for a batch of rays.
    Args:
        sigmas: (B, S, 1) density σ >= 0
        rgbs:   (B, S, 3) color in [0,1]
        step_size: float, ∆t between adjacent samples along ray

    Returns:
        rendered_colors: (B, 3)
    """
    # alpha_i = 1 - exp(-sigma_i * delta)
    # weight_i = T_i * alpha_i,   T_i = Π_{j<i}(1 - alpha_j)
    # C = Σ_i weight_i * rgb_i
    # Implemented in torch for backprop.

    # (B, S, 1)
    alphas = 1.0 - torch.exp(-sigmas * step_size)

    # (B, S) for cumulative products
    alphas = alphas.squeeze(-1)  # (B, S)

    # Compute transmittance T_i: cumprod over (1 - alpha), shifted by 1 with T_0=1
    # torch.cumprod is inclusive, so we shift by concatenating a leading 1
    one_minus_alpha = 1.0 - alphas  # (B, S)
    # T_prefix = [1, Π_{j<=i-1}(1-alpha_j)]
    T_prefix = torch.cumprod(
        torch.cat([torch.ones_like(one_minus_alpha[:, :1]), one_minus_alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]  # (B, S)

    weights = T_prefix * alphas  # (B, S)

    # Composite colors
    rendered = torch.sum(weights.unsqueeze(-1) * rgbs, dim=1)  # (B, 3)
    return rendered
