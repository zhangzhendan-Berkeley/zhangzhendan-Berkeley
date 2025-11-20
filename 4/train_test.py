# test_volrend_and_render.py
import torch
import numpy as np
from volume_render import volrend
from nerf_network import NeRF
from render_utils import render_one_view

def test_volrend_from_spec():
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volrend(sigmas, rgbs, step_size)

    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)
    print("[PASS] volrend matches the expected output.")

def smoke_render_one_view():
    # Dummy camera (identity) and small image to just test the pipeline runs
    H, W = 32, 32
    focal = 50.0
    K = np.array([[focal, 0, W/2],
                  [0, focal, H/2],
                  [0,     0,   1]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)

    model = NeRF(W=128, D=4, Lx=6, Ld=3).to("cuda" if torch.cuda.is_available() else "cpu")
    img = render_one_view(model, K, c2w, H, W, near=2.0, far=6.0, n_samples=32, chunk=1024)
    print("[SMOKE] render_one_view ok. Image stats:", img.shape, img.min(), img.max())

if __name__ == "__main__":
    test_volrend_from_spec()
    smoke_render_one_view()
