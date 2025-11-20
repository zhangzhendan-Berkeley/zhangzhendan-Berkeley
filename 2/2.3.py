import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_stack(img, levels):
    g_stk = [img.copy()]
    for i in range(levels):
        img = cv2.GaussianBlur(img, (25, 25), 50)
        g_stk.append(img)
    return g_stk

def laplacian_stack(img, levels):
    img = img.astype(np.float32)
    g_stk = [img.copy()]
    for i in range(levels):
        img = cv2.GaussianBlur(img, (25,25), 50)
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

# ================= Load Images ==================
apple = cv2.imread("apple.jpeg")
orange = cv2.imread("orange.jpeg")

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

show_stack(gas_apple, "Apple Gaussian Stack")
show_stack(gas_orange, "Orange Gaussian Stack")
show_stack(lap_apple, "Apple Laplacian Stack", gray=True)
show_stack(lap_orange, "Orange Laplacian Stack", gray=True)

# ================ Make Mask ===================
w_grad = 100
x_start = w//2 - w_grad//2
x_end = x_start + w_grad

mask = np.zeros((h, w), dtype=np.float32)
mask[:, :x_start] = 1
mask[:, x_end:] = 0
mask[:, x_start:x_end] = np.linspace(1, 0, w_grad, dtype=np.float32)

plt.figure(figsize=(8,2))
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.title("Base Mask (Gradient Window)")
plt.show()

# =============== Gaussian stack for Mask =================
def gaussian_stack_mask(mask, levels):
    g_stk = [mask.astype(np.float32)]
    for i in range(levels):
        blurred = cv2.GaussianBlur(g_stk[-1], (25,25), 50)
        g_stk.append(blurred.astype(np.float32))
    return g_stk

mask_stack = gaussian_stack_mask(mask, levels)

show_stack(mask_stack, "Mask Gaussian Stack", gray=True)

# =============== Blend Each Layer ===================
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
    if i == 0: plt.title("Apple Only")

    plt.subplot(levels, 3, i*3 + 2)
    plt.imshow(orange_vis, cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Orange Only")

    plt.subplot(levels, 3, i*3 + 3)
    plt.imshow(fused_vis, cmap='gray')
    plt.axis('off')
    if i == 0: plt.title("Blended Layer")

plt.suptitle("Each Laplacian Layer after Multi-Scale Mask Fusion")
plt.show()

# =============== Reconstruct ===================
final_img = np.zeros_like(blended_stack[0], dtype=np.float32)
for layer in blended_stack:
    final_img += layer.astype(np.float32)
final_img = np.clip(final_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(final_img)
plt.axis('off')
plt.title("Final Blended Image (Oraple)")
plt.show()
