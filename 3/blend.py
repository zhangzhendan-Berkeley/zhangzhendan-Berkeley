import numpy as np

def create_alpha_mask(h, w):
    """Create a feathering alpha mask that is 1 in the center and falls to 0 at borders."""
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    # distance from borders (left, right, top, bottom)
    dist_x = np.minimum(x, 1-x)
    dist_y = np.minimum(y, 1-y)
    mask = np.minimum(dist_x, dist_y)
    mask = mask / mask.max()  # normalize to [0,1]
    return mask

def blend_images(base_img, warped_img, base_mask, warped_mask):
    """Blend two images using weighted average with masks."""
    base_mask = base_mask[..., None]
    warped_mask = warped_mask[..., None]

    num = base_img.astype(np.float32) * base_mask + warped_img.astype(np.float32) * warped_mask
    denom = base_mask + warped_mask
    denom[denom==0] = 1e-6
    blended = (num / denom).astype(np.uint8)
    return blended

def compute_mosaic_canvas(im1, im2, H):
    """计算mosaic画布大小，以及平移矩阵T"""
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # 四个角点
    corners1 = np.array([[0,0,1],[w1,0,1],[0,h1,1],[w1,h1,1]], dtype=np.float32).T
    corners2 = np.array([[0,0,1],[w2,0,1],[0,h2,1],[w2,h2,1]], dtype=np.float32).T

    # warp 左图的角点
    warped_corners1 = H @ corners1
    warped_corners1 /= warped_corners1[2]

    # 合并所有角点
    all_corners = np.hstack((warped_corners1, corners2))
    x_min, y_min = np.floor(all_corners[:2].min(axis=1)).astype(int)
    x_max, y_max = np.ceil(all_corners[:2].max(axis=1)).astype(int)

    width, height = x_max - x_min, y_max - y_min

    # 平移矩阵 T
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float32)

    return (height, width), T


