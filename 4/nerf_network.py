# nerf_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -------------------------
# Positional Encoding (PE)
# -------------------------
class PositionalEncoding(nn.Module):
    """
    For each input x in R^C, output:
    [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{L-1} pi x), cos(2^{L-1} pi x)]
    Output dim = C * (2*L + 1)
    """
    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        # register buffers for frequencies
        freqs = 2.0 ** torch.arange(num_freqs).float() * torch.pi  # [L]
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        # original + sin/cos pairs
        return None  # filled in forward from input dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., C)
        return: (..., C * (2*L + 1))
        """
        # original term
        outs = [x]
        # sin/cos for each freq
        for f in self.freqs:
            outs.append(torch.sin(f * x))
            outs.append(torch.cos(f * x))
        return torch.cat(outs, dim=-1)

def pe_out_dim(input_dim: int, L: int) -> int:
    return input_dim * (2 * L + 1)

# -------------------------
# NeRF MLP (coarse/fine同构)
# -------------------------
class NeRF(nn.Module):
    """
    NeRF-style network with:
      - deeper coordinate tower with skip at layer 4
      - density (sigma) head from tower features (ReLU)
      - color head conditioned on view direction encoding (Sigmoid)
    Inputs:
      x: world coords (..., 3)
      d: view dirs   (..., 3), will be normalized inside
    """
    def __init__(
        self,
        W: int = 256,           # width
        D: int = 8,             # depth (layers)
        skips: Tuple[int, ...] = (4,),
        Lx: int = 10,           # PE freq for coords
        Ld: int = 4,            # PE freq for dirs (smaller)
    ):
        super().__init__()
        self.W, self.D, self.skips, self.Lx, self.Ld = W, D, skips, Lx, Ld

        in_x = pe_out_dim(3, Lx)   # 3 * (2*Lx+1)
        in_d = pe_out_dim(3, Ld)   # 3 * (2*Ld+1)

        # coordinate tower with skip at layer 4
        layers = []
        dim_in = in_x
        for i in range(D):
            if i in skips:
                dim_in = in_x + W  # concat input PE later
            layers.append(nn.Linear(dim_in, W))
            dim_in = W
        self.layers = nn.ModuleList(layers)

        # sigma (density) head
        self.sigma_head = nn.Linear(W, 1)

        # feature for color branch
        self.feature_head = nn.Linear(W, W)

        # color head: features (+ dir PE) -> 128 -> rgb
        self.color_fc = nn.Sequential(
            nn.Linear(W + in_d, 128),
            nn.ReLU(True),
            nn.Linear(128, 3)
        )

        # encoders
        self.pe_x = PositionalEncoding(Lx)
        self.pe_d = PositionalEncoding(Ld)

        # init
        self._init_weights()

    def _init_weights(self):
        def init_lin(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_lin)

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        x: (..., 3) world coords
        d: (..., 3) view dirs
        returns:
          sigma: (..., 1)  (non-negative)
          rgb:   (..., 3)  (in [0,1])
        """
        orig_shape = x.shape[:-1]  # keep for reshape back
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)

        # normalize dirs
        d = d / (torch.linalg.norm(d, dim=-1, keepdim=True) + 1e-9)

        # encodings
        x_enc = self.pe_x(x)      # [N, in_x]
        d_enc = self.pe_d(d)      # [N, in_d]

        # coordinate tower with skip
        h = x_enc
        idx_layer_in = 0
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, x_enc], dim=-1)
            h = layer(h)
            h = F.relu(h, inplace=True)

        # sigma & features
        sigma = F.relu(self.sigma_head(h), inplace=False)  # non-negative
        feat  = F.relu(self.feature_head(h), inplace=True)

        # color branch conditioned on view dir
        h_col = torch.cat([feat, d_enc], dim=-1)
        rgb   = torch.sigmoid(self.color_fc(h_col))

        # reshape back
        sigma = sigma.view(*orig_shape, 1)
        rgb   = rgb.view(*orig_shape, 3)
        return sigma, rgb

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    import torch
    model = NeRF(W=256, D=8, Lx=10, Ld=4)
    print("Params:", model.num_params)

    # batch of 10 rays × 64 samples
    B, S = 10, 64
    x = torch.randn(B, S, 3)
    d = torch.randn(B, S, 3)
    sigma, rgb = model(x, d)

    print("sigma:", sigma.shape, sigma.min().item(), sigma.max().item())  # (10,64,1) >= 0
    print("rgb:",   rgb.shape,   rgb.min().item(),   rgb.max().item())    # (10,64,3) ~ [0,1]
