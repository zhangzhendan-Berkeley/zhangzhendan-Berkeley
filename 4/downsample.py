import os
from PIL import Image

# ============================
# 用户配置
# ============================
# 输入图片或文件夹路径
INPUT_PATH = "scan/"   # 可改成文件夹路径，例如 "scan/"
# 输出文件夹
OUTPUT_DIR = "scan_downsampled"
# 降采样后的最长边长度（像素）
MAX_SIZE = 1000
# 文件名前缀（可选）
PREFIX = "ds_"

# ============================
# 函数定义
# ============================
def downsample_and_save(image_path, out_dir, max_size=256, prefix="ds_"):
    """对单张图片降采样并保存"""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] 无法读取 {image_path}: {e}")
        return

    w, h = img.size
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(image_path)
    save_path = os.path.join(out_dir, prefix + fname)
    img_resized.save(save_path)
    print(f"[INFO] {fname}  →  {new_w}×{new_h}  saved to  {save_path}")


# ============================
# 主程序
# ============================
if __name__ == "__main__":
    if os.path.isdir(INPUT_PATH):
        # 批量处理整个文件夹
        for fname in os.listdir(INPUT_PATH):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(INPUT_PATH, fname)
                downsample_and_save(full_path, OUTPUT_DIR, MAX_SIZE, PREFIX)
    elif os.path.isfile(INPUT_PATH):
        # 单张图片
        downsample_and_save(INPUT_PATH, OUTPUT_DIR, MAX_SIZE, PREFIX)
    else:
        print(f"[ERROR] 未找到路径: {INPUT_PATH}")
