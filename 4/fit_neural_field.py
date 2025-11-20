import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================
# 0. 位置编码 (Sinusoidal Positional Encoding)
# ============================================================
def positional_encoding(x, L=10):
    """x: [N, 2] normalized coords in [0, 1]"""
    enc = [x]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            enc.append(fn((2.0 ** i) * np.pi * x))
    return torch.cat(enc, dim=-1)


# ============================================================
# 1. MLP 定义
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # 输出 RGB ∈ [0, 1]
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# 2. 随机采样像素 dataloader
# ============================================================
def sample_pixels(img, num_samples):
    """img: numpy array [H, W, 3] (0-255)"""
    H, W, _ = img.shape
    xs = np.random.randint(0, W, num_samples)
    ys = np.random.randint(0, H, num_samples)
    coords = np.stack([xs / W, ys / H], axis=1)  # normalize
    colors = img[ys, xs] / 255.0                 # normalize
    return torch.tensor(coords, dtype=torch.float32), \
           torch.tensor(colors, dtype=torch.float32)


# ============================================================
# 3. PSNR 计算函数
# ============================================================
def psnr_from_mse(mse):
    return -10.0 * torch.log10(mse)


# ============================================================
# 4. 主训练逻辑
# ============================================================
def train_neural_field(
    image_path,
    max_freq=10,
    hidden_dim=256,
    lr=1e-2,
    num_iters=2000,
    batch_size=10000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # 读取图像
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    H, W, _ = img.shape
    print(f"[INFO] Image loaded: {image_path}, shape = {img.shape}")

    # 模型与优化器
    in_dim = 2 * (2 * max_freq + 1)
    model = MLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    psnr_list = []
    save_dir = "training_vis"
    os.makedirs(save_dir, exist_ok=True)

    # 训练循环
    for it in range(1, num_iters + 1):
        coords, colors = sample_pixels(img, batch_size)
        coords, colors = coords.to(device), colors.to(device)

        pe = positional_encoding(coords, L=max_freq).to(device)
        pred = model(pe)
        loss = criterion(pred, colors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 500 == 0 or it == 1:
            psnr = psnr_from_mse(loss)
            psnr_list.append(psnr.item())
            print(f"Iter {it:04d} | Loss={loss.item():.6f} | PSNR={psnr.item():.2f} dB")

            # 每100次保存重建图像
            with torch.no_grad():
                xs = torch.linspace(0, 1, W)
                ys = torch.linspace(0, 1, H)
                grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
                coords_full = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(device)
                preds = model(positional_encoding(coords_full, L=max_freq)).reshape(H, W, 3)
                img_pred = preds.cpu().numpy()

                plt.figure(figsize=(6, 3))
                plt.subplot(1, 2, 1)
                plt.title("GT")
                plt.imshow(img.astype(np.uint8))
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.title(f"Iter {it}")
                plt.imshow(np.clip(img_pred, 0, 1))
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"{save_dir}/iter_{it:04d}_freq{max_freq}_w{hidden_dim}.png")
                plt.close()

    # ========== 保存最终输出图像 ==========
    with torch.no_grad():
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        coords_full = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(device)
        preds = model(positional_encoding(coords_full, L=max_freq)).reshape(H, W, 3)
        img_pred = preds.cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(np.clip(img_pred, 0, 1))
        plt.axis("off")
        plt.title(f"Final freq={max_freq}, width={hidden_dim}")
        final_path = f"{save_dir}/final_freq{max_freq}_w{hidden_dim}.png"
        plt.savefig(final_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[SAVED] Final output saved to {final_path}")

    # 保存 PSNR 曲线
    plt.figure()
    plt.plot(np.arange(len(psnr_list)) * 100, psnr_list)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Training Curve (freq={max_freq}, width={hidden_dim})")
    plt.tight_layout()
    curve_path = f"{save_dir}/psnr_curve_freq{max_freq}_w{hidden_dim}.png"
    plt.savefig(curve_path)
    plt.close()
    print(f"[SAVED] PSNR curve saved to {curve_path}")

    return psnr_list


# ============================================================
# 5. 主入口
# ============================================================
if __name__ == "__main__":
    image_path = "scan_downsampled/ds_20251106-175832(12).jpg"  # 降采样后的图片

    configs = [
        (4, 64),
        (4, 256),
        (10, 64),
        (10, 256),
    ]

    for freq, width in configs:
        print(f"\n========== Training: freq={freq}, width={width} ==========")
        psnr_list = train_neural_field(
            image_path=image_path,
            max_freq=freq,
            hidden_dim=width,
            lr=1e-2,
            num_iters=10000,
            batch_size=10000,
        )
