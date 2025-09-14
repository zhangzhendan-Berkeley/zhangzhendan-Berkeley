import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_stack(img, levels):
    g_stk = [img.copy()]
    for i in range(levels):
        img = cv2.GaussianBlur(img, (51, 51), 100)
        g_stk.append(img)
    return g_stk

def laplacian_stack(img, levels):
    img = img.astype(np.float32)
    g_stk = [img.copy()]
    for i in range(levels):
        img = cv2.GaussianBlur(img, (51,51), 100)
        g_stk.append(img.astype(np.float32))
    l_stk = []
    for i in range(levels):
        lap = g_stk[i] - g_stk[i+1]
        l_stk.append(lap)
    l_stk.append(g_stk[-1])
    return l_stk

def show_stack(stack, title, gray=False):
    levels = len(stack)
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(stack):
        plt.subplot(1, levels, i+1)
        if gray:
            max_abs = max(abs(img.min()), abs(img.max()))
            img_vis = 127 + (img / max_abs) * 127
            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
            plt.imshow(img_vis, cmap='gray')
        else:
            plt.imshow(img.astype(np.uint8))
        plt.axis('off')
        plt.title(f'Level {i}')
    plt.suptitle(title)
    plt.show()

apple = cv2.imread("mouth.jpg")
orange = cv2.imread("hand.jpg")

h = min(apple.shape[0], orange.shape[0])
w = min(apple.shape[1], orange.shape[1])
apple = cv2.resize(apple, (w, h))
orange = cv2.resize(orange, (w, h))

apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)
levels = 6

gas_apple = gaussian_stack(apple, levels)
gas_orange = gaussian_stack(orange, levels)
lap_apple = laplacian_stack(apple, levels)
lap_orange = laplacian_stack(orange, levels)

show_stack(gas_apple, "Gaussian Stack 1")
show_stack(gas_orange, "Gaussian Stack 2")
show_stack(lap_apple, "Laplacian Stack 1", gray=True)
show_stack(lap_orange, "Laplacian Stack 2", gray=True)

def create_center_mask(shape, r_inner=30, r_outer=60, mode='linear'):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    mask = np.zeros_like(d, dtype=np.float32)

    if mode == 'linear':
        mask = np.clip((r_outer - d) / (r_outer - r_inner), 0, 1)
    elif mode == 'gaussian':
        sigma = (r_outer - r_inner) / 2
        center = (r_inner + r_outer) / 2
        mask = np.exp(-((d - r_inner) ** 2) / (2 * sigma ** 2))
        mask[d < r_inner] = 1
        mask[d > r_outer] = 0
    else:
        raise ValueError("mode must be 'linear' or 'gaussian'")

    mask[d <= r_inner] = 1
    mask[d >= r_outer] = 0

    return mask

def create_elliptical_mask(shape, r_inner=0.3, r_outer=0.5, a=1.0, b=1.0, mode='linear'):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2

    dx = (x - cx) / (a * w / 2)
    dy = (y - cy) / (b * h / 2)
    d = np.sqrt(dx**2 + dy**2)

    mask = np.zeros_like(d, dtype=np.float32)

    if mode == 'linear':
        mask = np.clip((r_outer - d) / (r_outer - r_inner), 0, 1)
    elif mode == 'gaussian':
        sigma = (r_outer - r_inner) / 2
        mask = np.exp(-((d - r_inner) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError("mode must be 'linear' or 'gaussian'")

    mask[d <= r_inner] = 1
    mask[d >= r_outer] = 0

    return mask

mask_gray = create_elliptical_mask((h, w), r_inner=0.2, r_outer=0.5, a=1, b=0.6, mode='linear')

mask_stack = gaussian_stack(mask_gray, levels)

mask_stack = [m.astype(np.float32) / m.max() for m in mask_stack]

plt.figure(figsize=(15,3))
for i, m in enumerate(mask_stack):
    plt.subplot(1, levels+1, i+1)
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.title(f'Mask {i}')
plt.suptitle("Gaussian Mask Stack")
plt.show()

blended_stack = []

levels = len(lap_apple)
plt.figure(figsize=(15, levels*3))

for i, (l1, l2, m) in enumerate(zip(lap_apple, lap_orange, mask_stack)):
    if len(l1.shape) == 3:
        m3 = np.repeat(m[:, :, np.newaxis], 3, axis=2)
    else:
        m3 = m

    blended_layer = l1 * m3 + l2 * (1 - m3)
    blended_stack.append(blended_layer)

    # 可视化
    max_abs = max(abs(blended_layer.min()), abs(blended_layer.max()))
    fused_vis = 127 + (blended_layer / max_abs) * 127
    fused_vis = np.clip(fused_vis, 0, 255).astype(np.uint8)

    apple_only = l1 * m3
    apple_vis = 127 + (apple_only / max_abs) * 127
    apple_vis = np.clip(apple_vis, 0, 255).astype(np.uint8)

    orange_only = l2 * (1 - m3)
    orange_vis = 127 + (orange_only / max_abs) * 127
    orange_vis = np.clip(orange_vis, 0, 255).astype(np.uint8)

    plt.subplot(levels, 3, i*3 + 1)
    plt.imshow(apple_vis, cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("1 Only")

    plt.subplot(levels, 3, i*3 + 2)
    plt.imshow(orange_vis, cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("2 Only")

    plt.subplot(levels, 3, i*3 + 3)
    plt.imshow(fused_vis, cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Blended Layer")

plt.suptitle("Each Laplacian Layer with Mask Stack Fusion")
plt.show()

# === 重建最终图像 ===
final_img = np.zeros_like(blended_stack[0], dtype=np.float32)
for layer in blended_stack:
    final_img += layer.astype(np.float32)
final_img = np.clip(final_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(final_img)
plt.axis('off')
plt.title("Final Blended Image with Mask Stack")
plt.show()

