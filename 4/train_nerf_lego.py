import os, time, math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
from PIL import Image
import imageio.v2 as imageio

from rays_geometry import pixel_to_ray
from sample_along_rays import sample_along_rays
from volume_render import volrend
from render_utils import render_one_view
from nerf_network import NeRF

# -------------------------
# Utils
# -------------------------
def to_torch32(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def psnr_from_mse_t(mse_t: torch.Tensor) -> float:
    return (-10.0 * torch.log10(mse_t)).item()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# -------------------------
# Ray sampler (global)
# -------------------------
@torch.no_grad()
def sample_random_rays_global(
    images: np.ndarray,   # (N,H,W,3) in [0,1]
    K: np.ndarray,        # (3,3)
    c2ws: np.ndarray,     # (N,4,4)
    n_rays: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从所有训练图像里全局随机采样 n_rays 条光线
    返回:
      rays_o: (n_rays,3), rays_d: (n_rays,3), target_rgb: (n_rays,3)
    """
    N, H, W, _ = images.shape
    # 随机选择图像 & 像素
    img_ids = np.random.randint(0, N, size=(n_rays,))
    us = np.random.randint(0, W, size=(n_rays,))
    vs = np.random.randint(0, H, size=(n_rays,))

    # 目标颜色
    target = images[img_ids, vs, us]   # (n_rays,3)

    # 统一 K（lego 恒定），逐图批量生成光线
    K_t = to_torch32(K, device)
    uvs = np.stack([us + 0.5, vs + 0.5], axis=1)
    uvs_t = torch.tensor(uvs, dtype=torch.float32, device=device)

    rays_o = torch.empty((n_rays, 3), dtype=torch.float32, device=device)
    rays_d = torch.empty((n_rays, 3), dtype=torch.float32, device=device)

    # 按图像 id 分组，减少重复变换
    unique_ids = np.unique(img_ids)
    offset_mask = np.zeros(n_rays, dtype=bool)
    for uid in unique_ids:
        mask = (img_ids == uid)
        idxs = np.where(mask)[0]
        c2w_t = to_torch32(c2ws[uid], device)
        # 选出该图像的像素
        ray_o, ray_d = pixel_to_ray(K_t, c2w_t, uvs_t[idxs])
        rays_o[idxs] = ray_o
        rays_d[idxs] = ray_d
        offset_mask[idxs] = True

    assert offset_mask.all()
    target_t = torch.tensor(target, dtype=torch.float32, device=device)
    return rays_o, rays_d, target_t

# -------------------------
# Train / Val helpers
# -------------------------
@torch.no_grad()
def validate_psnr(
    model: NeRF,
    images_val: np.ndarray,
    K: np.ndarray,
    c2ws_val: np.ndarray,
    n_imgs: int = 6,
    out_dir: str = "results/val_snaps",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, List[str]]:
    """渲染若干验证视角并计算平均 PSNR，保存可视化快照"""
    ensure_dir(out_dir)
    H, W = images_val.shape[1:3]
    picks = list(range(min(n_imgs, images_val.shape[0])))

    psnrs = []
    saved = []
    for i in picks:
        gt = images_val[i]
        pred = render_one_view(
            model, K, c2ws_val[i], H, W,
            near=2.0, far=6.0, n_samples=64, chunk=8192, device=device
        )
        # MSE & PSNR
        mse = np.mean((pred - gt) ** 2) + 1e-12
        psnr = -10.0 * math.log10(mse)
        psnrs.append(psnr)

        # 保存叠放图
        canvas = np.concatenate([gt, pred], axis=1)  # 左GT 右Pred
        path = os.path.join(out_dir, f"val_{i:02d}.png")
        Image.fromarray((canvas * 255).astype(np.uint8)).save(path)
        saved.append(path)

    return float(np.mean(psnrs)), saved

# -------------------------
# Main training
# -------------------------
def main():
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = "results"
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, "train_snaps"))
    ensure_dir(os.path.join(results_dir, "val_snaps"))

    # Hyperparams
    iters         = 1000            # 课程参考：1000 steps 可达 23+ PSNR
    rays_per_iter = 10_000          # 每步采样光线数
    n_samples     = 64              # 每条光线采样点数
    near, far     = 2.0, 6.0
    lr            = 5e-4            # 课程建议
    log_interval  = 50
    val_interval  = 100

    # Load lego data
    data = np.load("lego_200x200.npz")
    images_train = data["images_train"] / 255.0  # (100,200,200,3)
    c2ws_train   = data["c2ws_train"]
    images_val   = data["images_val"]   / 255.0  # (10,200,200,3)
    c2ws_val     = data["c2ws_val"]
    c2ws_test    = data["c2ws_test"]            # (60,4,4)
    focal        = float(data["focal"])
    H, W = images_train.shape[1:3]
    K = np.array([[focal, 0, W/2],
                  [0, focal, H/2],
                  [0, 0, 1   ]], dtype=np.float32)

    # Model
    model = NeRF(W=256, D=8, Lx=10, Ld=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Logs
    train_losses = []
    val_psnrs = []

    print(f"[INFO] Start training on {device} | steps={iters} | rays/it={rays_per_iter}")

    t0 = time.time()
    for it in range(1, iters + 1):
        model.train()
        # 1) sample random rays globally
        rays_o, rays_d, target_rgb = sample_random_rays_global(
            images_train, K, c2ws_train, rays_per_iter, device
        )
        # 2) march along rays
        pts, _ = sample_along_rays(rays_o, rays_d,
                                   n_samples=n_samples, near=near, far=far, perturb=True)
        # 3) network forward
        sigma, rgb = model(pts, rays_d.unsqueeze(1).expand_as(pts))
        # 4) volume render
        step_size = (far - near) / n_samples
        pred_rgb = volrend(sigma, rgb, step_size=step_size)  # (B,3)
        # 5) loss
        loss = criterion(pred_rgb, target_rgb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if it % log_interval == 0 or it == 1:
            psnr = psnr_from_mse_t(loss.detach())
            elapsed = time.time() - t0
            print(f"[{it:04d}/{iters}] loss={loss.item():.6f} | psnr={psnr:.2f} dB | {elapsed:.1f}s")

        # 验证 & 可视化
        if it % val_interval == 0 or it == iters:
            model.eval()
            # 渲染 1 张训练图作为进度快照
            snap_pred = render_one_view(
                model, K, c2ws_train[0], H, W,
                near=near, far=far, n_samples=n_samples, chunk=8192, device=device
            )
            snap_canvas = np.concatenate([images_train[0], snap_pred], axis=1)
            Image.fromarray((snap_canvas * 255).astype(np.uint8)).save(
                os.path.join(results_dir, "train_snaps", f"iter_{it:04d}.png")
            )
            # 评估验证 PSNR（6张）
            mean_psnr, saved_paths = validate_psnr(
                model, images_val, K, c2ws_val, n_imgs=6,
                out_dir=os.path.join(results_dir, "val_snaps"), device=device
            )
            val_psnrs.append((it, mean_psnr))
            print(f"  [val] PSNR(mean over 6) = {mean_psnr:.2f} dB")

    # 保存曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Train MSE")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "train_loss_curve.png"))
    plt.close()

    if len(val_psnrs) > 0:
        its, ps = zip(*val_psnrs)
        plt.figure()
        plt.plot(list(its), list(ps))
        plt.xlabel("Iteration")
        plt.ylabel("PSNR (dB)")
        plt.title("Validation PSNR (6 images)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "val_psnr_curve.png"))
        plt.close()

    # 渲染测试相机轨迹视频 (spherical)
    print("[INFO] Rendering c2ws_test video ...")
    frames = []
    for i, c2w in enumerate(c2ws_test):
        img = render_one_view(
            model, K, c2w, H, W,
            near=near, far=far, n_samples=n_samples, chunk=8192, device=device
        )
        frames.append((img * 255).astype(np.uint8))
    gif_path = os.path.join(results_dir, "lego_spin.gif")
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"[DONE] Saved video: {gif_path}")

if __name__ == "__main__":
    main()
