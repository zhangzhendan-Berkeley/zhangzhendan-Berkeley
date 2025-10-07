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

print("PYTORCH_CUDA_ALLOC_CONF =", os.getenv("PYTORCH_CUDA_ALLOC_CONF"))

torch.cuda.empty_cache()

# =========================
# Helper / Utils
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def to_torch32(x, device): return torch.tensor(x, dtype=torch.float32, device=device)
def psnr_from_mse_t(mse_t: torch.Tensor) -> float: return (-10.0 * torch.log10(mse_t + 1e-12)).item()

def to_uint8_rgb(img01: np.ndarray, assume_bgr: bool = True) -> np.ndarray:
    """
    img01: [H,W,3] 浮点 0..1
    assume_bgr=True 表示 img01 的通道顺序是 BGR，需要转为 RGB 才能给 PIL/imageio 看
    """
    img = np.clip(img01, 0.0, 1.0)
    if assume_bgr:
        img = img[..., ::-1]  # BGR -> RGB
    return (img * 255.0 + 0.5).astype(np.uint8)

def save_png(path: str, img01: np.ndarray, assume_bgr: bool = True):
    Image.fromarray(to_uint8_rgb(img01, assume_bgr=assume_bgr)).save(path)

def look_at_origin(pos):
    pos = np.asarray(pos, dtype=np.float32)
    forward = -pos / (np.linalg.norm(pos) + 1e-9)
    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(up, forward); right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right; c2w[:3, 1] = up; c2w[:3, 2] = forward; c2w[:3, 3] = pos
    return c2w

def rot_x(phi):
    return np.array([
        [math.cos(phi), -math.sin(phi), 0, 0],
        [math.sin(phi),  math.cos(phi), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ], dtype=np.float32)

@torch.no_grad()
def sample_random_rays_global(images, K, c2ws, n_rays, device):
    """
    从所有训练图像里全局随机采样 n_rays 条光线
    返回: rays_o, rays_d, target_rgb (n_rays,3)
    """
    N, H, W, _ = images.shape
    img_ids = np.random.randint(0, N, size=(n_rays,))
    us = np.random.randint(0, W, size=(n_rays,))
    vs = np.random.randint(0, H, size=(n_rays,))
    target = images[img_ids, vs, us]   # (n_rays,3)

    K_t = to_torch32(K, device)
    uvs = np.stack([us + 0.5, vs + 0.5], axis=1)
    uvs_t = torch.tensor(uvs, dtype=torch.float32, device=device)

    rays_o = torch.empty((n_rays, 3), dtype=torch.float32, device=device)
    rays_d = torch.empty((n_rays, 3), dtype=torch.float32, device=device)

    unique_ids = np.unique(img_ids)
    for uid in unique_ids:
        mask = (img_ids == uid)
        idxs = np.where(mask)[0]
        c2w_t = to_torch32(c2ws[uid], device)
        ro, rd = pixel_to_ray(K_t, c2w_t, uvs_t[idxs])
        rays_o[idxs] = ro; rays_d[idxs] = rd

    target_t = torch.tensor(target, dtype=torch.float32, device=device)
    return rays_o, rays_d, target_t

@torch.no_grad()
def validate_psnr(model, images_val, K, c2ws_val, near, far, n_samples, out_dir, device, max_imgs=6):
    ensure_dir(out_dir)
    H, W = images_val.shape[1:3]
    picks = list(range(min(max_imgs, images_val.shape[0])))

    psnrs = []
    for i in picks:
        gt = images_val[i]
        pred = render_one_view(
            model, K, c2ws_val[i], H, W,
            near=near, far=far, n_samples=n_samples,
            chunk=8192, device=device
        )

        # ---- 数值保护：防止 NaN / Inf / 越界 ----
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        mse = np.mean((pred - gt) ** 2) + 1e-12
        psnr = -10.0 * math.log10(mse)
        psnrs.append(psnr)

        # 拼接 GT 与预测结果 (横向对比)
        canvas = np.concatenate([gt, pred], axis=1)
        canvas = np.clip(canvas, 0.0, 1.0)  # 再保险
        save_png(os.path.join(out_dir, f"val_{i:02d}.png"), canvas, assume_bgr=True)

    return float(np.mean(psnrs)) if psnrs else 0.0

@torch.no_grad()
def offline_render_all(checkpoints_dir, images_train, c2ws_train, K, H, W, near, far, n_samples, c2ws_test, device):
    # load all checkpoints
    ckpts = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")])
    out_snap = os.path.join("results_custom", "train_snaps")
    ensure_dir(out_snap)

    print("[INFO] Start offline rendering...")

    for ck in ckpts:
        print("[INFO] Rendering snapshot for", ck)
        data = torch.load(os.path.join(checkpoints_dir, ck), map_location=device)

        model = NeRF(W=512, D=8, Lx=12, Ld=4).to(device)
        model.load_state_dict(data["model"])
        model.eval()

        iter_id = data["iter"]

        # ---- render train[0] for snapshot ----
        pred = render_one_view(
            model, K, c2ws_train[0], H, W,
            near=near, far=far, n_samples=n_samples, chunk=8192, device=device
        )
        gt = images_train[0]
        canvas = np.concatenate([gt, pred], axis=1)
        save_png(os.path.join(out_snap, f"iter_{iter_id:05d}.png"), canvas, assume_bgr=True)

    # ---- render GIF using final checkpoint ----
    print("[INFO] Rendering GIF using last checkpoint:", ckpts[-1])
    data = torch.load(os.path.join(checkpoints_dir, ckpts[-1]), map_location=device)

    model = NeRF(W=512, D=8, Lx=12 , Ld=4).to(device)
    model.load_state_dict(data["model"])
    model.eval()

    frames = []
    for c2w in c2ws_test:
        img = render_one_view(model, K, c2w, H, W,
                              near=near, far=far, n_samples=n_samples, chunk=8192, device=device)
        frames.append(to_uint8_rgb(img, assume_bgr=True))

    gif_path = os.path.join("results_custom", "final_orbit.gif")
    imageio.mimsave(gif_path, frames, fps=15)
    print("[DONE] Saved GIF →", gif_path)

# =========================
# Main (custom data)
# =========================
def main():
    # -------------------------
    # Config
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

    results_dir = "results_custom"
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, "train_snaps"))

    # NeRF 训练参数
    near, far = 0.001, 0.5
    n_samples = 128
    rays_per_iter = 5000
    iters = 40000
    lr = 5e-4
    log_interval = 50
    snapshot_interval = 8000   # ✅ 每 2000 轮输出一张训练渲染结果

    # -------------------------
    # Load dataset
    # -------------------------
    data = np.load("dataset/my_data.npz")
    images_train = data["images_train"] / 255.0
    c2ws_train   = data["c2ws_train"]
    K            = data["K"]
    H, W = images_train.shape[1:3]

    # -------------------------
    # Orbit trajectory
    # -------------------------
    c0 = c2ws_train[1].astype(np.float32)
    x0, y0, z0 = c0[0,3], c0[1,3], c0[2,3]
    R = float(np.sqrt(x0*x0 + y0*y0))
    if R < 1e-6: R = 0.3

    NUM_SAMPLES = 30
    c2ws_test = []
    for phi in np.linspace(0, 2*np.pi, NUM_SAMPLES, endpoint=False):
        cam_pos = np.array([R*np.cos(phi), R*np.sin(phi), z0], dtype=np.float32)
        target  = np.array([0,0,0], dtype=np.float32)
        up = np.array([0,0,-1], dtype=np.float32)

        forward = target - cam_pos; forward /= np.linalg.norm(forward)+1e-9
        right   = np.cross(up, forward); right /= np.linalg.norm(right)+1e-9
        true_up = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3,0] = right
        c2w[:3,1] = true_up
        c2w[:3,2] = forward
        c2w[:3,3] = cam_pos
        c2ws_test.append(c2w)
    c2ws_test = np.stack(c2ws_test, axis=0)

    print(f"[INFO] Test trajectory: {len(c2ws_test)} poses")

    # -------------------------
    # Model
    # -------------------------
    model = NeRF(W=512, D=8, Lx=12, Ld=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    # -------------------------
    # Train
    # -------------------------
    train_losses = []
    train_psnrs = []
    checkpoints_dir = os.path.join(results_dir, "ckpts")
    ensure_dir(checkpoints_dir)

    for it in range(1, iters + 1):
        model.train()
        rays_o, rays_d, target_rgb = sample_random_rays_global(
            images_train, K, c2ws_train, rays_per_iter, device
        )
        pts, _ = sample_along_rays(
            rays_o, rays_d, n_samples=n_samples,
            near=near, far=far, perturb=True
        )

        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
            sigma, rgb = model(pts, rays_d.unsqueeze(1).expand_as(pts))
            step = (far - near) / n_samples
            pred_rgb = volrend(sigma, rgb, step_size=step)
            loss = criterion(pred_rgb, target_rgb)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
        psnr = psnr_from_mse_t(loss.detach())
        train_psnrs.append(psnr)

        if it % log_interval == 0 or it == 1:
            print(f"[{it}/{iters}] loss={loss.item():.6f} | psnr={psnr:.2f} dB")

        # ✅ 只保存模型，不渲染任何图
        if it % snapshot_interval == 0 or it == iters:
            ckpt_path = os.path.join(checkpoints_dir, f"iter_{it:05d}.pth")
            torch.save({
                "iter": it,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)

    # -------------------------
    # Curves
    # -------------------------
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "train_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_psnrs)
    plt.xlabel("Iteration")
    plt.ylabel("Train PSNR (dB)")
    plt.title("Training PSNR")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "train_psnr_curve.png"))
    plt.close()

    offline_render_all("results_custom/ckpts",
                       images_train, c2ws_train,
                       K, H, W, near, far, n_samples,
                       c2ws_test, device)





if __name__ == "__main__":
    main()

