
import numpy as np
from skimage.feature import corner_harris, peak_local_max


def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    # print(im.shape)
    h = corner_harris(im, k=0.04, sigma=1)
    # print(h.shape)
    coords = peak_local_max(h, min_distance=5, threshold_rel=0.01)
    # print(coords.shape)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask]
    return h, coords

def anms(coords, strengths, N=500, robust_coef=0.9):
    """
    Adaptive Non-Maximal Suppression (ANMS).

    Args:
        coords: (N_all, 2) array of (row, col) coordinates of candidate corners.
        strengths: (N_all,) array of corner response values aligned with coords (higher = stronger).
        N: how many corners to return (top N by suppression radius).
        robust_coef: factor in (0,1]; consider j "stronger" than i if strengths[j] > strengths[i] * robust_coef.
                     Typical values: 0.9 (more permissive), 1.0 (strict).
    Returns:
        selected_coords: (M,2) array of chosen coords (M = min(N, N_all)), in (row, col) format.
        radii: (N_all,) array of suppression radii for all input coords.
    Note: O(n^2) naive implementation.
    """

    coords = np.asarray(coords)
    strengths = np.asarray(strengths)
    assert coords.shape[0] == strengths.shape[0], "coords and strengths must align"
    # print(coords.shape[0])
    n_all = coords.shape[0]
    if n_all == 0:
        return coords.copy(), np.array([])

    # Precompute pairwise squared distances (could be memory heavy for huge n)
    # We'll compute on the fly to keep memory moderate.
    radii = np.zeros(n_all, dtype=float)
    # image diagonal fallback for points with no stronger neighbors
    max_possible = np.sqrt((coords[:,0].max() - coords[:,0].min())**2 + (coords[:,1].max() - coords[:,1].min())**2)

    # For each corner i, find min distance to any corner j that is "stronger"
    for i in range(n_all):
        si = strengths[i]
        # indices of corners considered "stronger"
        stronger_idx = np.where(strengths > si * robust_coef)[0]
        if stronger_idx.size == 0:
            # no stronger corner -> give it a large radius
            radii[i] = max_possible
            continue
        # compute Euclidean distances to those stronger corners
        diffs = coords[stronger_idx] - coords[i]  # shape (k,2)
        dists = np.sqrt((diffs[:,0]**2) + (diffs[:,1]**2))
        radii[i] = dists.min()

    # pick top N by radii (larger r means more isolated/valuable)
    order = np.argsort(-radii)  # descending by radius
    pick = order[:min(N, n_all)]
    selected_coords = coords[pick]

    return selected_coords, radii


# ---------- 可视化辅助 ----------
def overlay_corners_on_image(img, corners, title="Corners", show_n=None, color=(1,0,0), marker='x', s=30):
    """
    img: BGR or grayscale array (if BGR, will convert to RGB for matplotlib).
    corners: (N,2) array (row, col)
    show_n: int or None, if int will randomly sample up to show_n points to draw (avoid overplot)
    color: matplotlib color tuple or string (default red)
    """
    if img.ndim == 3:
        plot_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        plot_img = img

    plt.figure(figsize=(8,8))
    if plot_img.ndim == 2:
        plt.imshow(plot_img, cmap='gray')
    else:
        plt.imshow(plot_img)
    if corners.size > 0:
        if show_n is not None and corners.shape[0] > show_n:
            idx = np.random.choice(corners.shape[0], show_n, replace=False)
            display = corners[idx]
        else:
            display = corners
        plt.scatter(display[:,1], display[:,0], c=[color], s=s, marker=marker)
    plt.title(title + f" ({corners.shape[0]} points)")
    plt.axis('off')
    plt.show()

def extract_descriptors(img, coords, patch_size=40, small_size=8):
    """
    Extract axis-aligned 8x8 descriptors from 40x40 windows.
    Args:
        img: grayscale image (float32, range [0,1])
        coords: Nx2 array of (x, y)
        patch_size: 40 (window size)
        small_size: 8 (final descriptor size)
    Returns:
        descriptors: Nx64
        valid_coords: Nx2 (coords that didn't go out of bounds)
    """
    half = patch_size // 2
    descriptors = []
    valid_coords = []

    for (y, x) in coords.astype(int):
        if (y - half < 0 or y + half >= img.shape[0] or
                x - half < 0 or x + half >= img.shape[1]):
            continue  # skip border

        patch = img[y - half:y + half, x - half:x + half]
        patch_small = cv2.resize(patch, (small_size, small_size), interpolation=cv2.INTER_AREA)

        mean = np.mean(patch_small)
        std = np.std(patch_small)
        desc = (patch_small - mean) / (std + 1e-5)
        descriptors.append(desc.flatten())
        valid_coords.append((y, x))

    return np.array(valid_coords), np.array(descriptors)

def preprocessing(img, N=500):
    h, coords = get_harris_corners(img)
    strengths = np.array([h[r, c] for r, c in coords])
    selected, radii = anms(coords, strengths, N=N, robust_coef=0.9)
    coords, desc = extract_descriptors(img, selected)
    return coords, desc

def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ---------- 读图和点 ----------
    img = cv2.imread("room_left.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, coords = get_harris_corners(img)

    # plt.figure(figsize=(12, 6))
    #
    # # 左图
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.scatter(coords[:, 1], coords[:, 0], s=5, c='red', marker='x')
    # plt.title(f'Left Image Corners ({len(coords)} points)')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    strengths = np.array([h[r, c] for r, c in coords])

    # ANMS: 选出 top N 点
    N_want = 500
    selected, radii = anms(coords, strengths, N=N_want, robust_coef=0.9)
    print("selected:", selected.shape[0])

    # # 可视化：全部候选 vs ANMS 结果覆盖
    # overlay_corners_on_image(img, coords, title="All Harris Candidates (before ANMS)", show_n=1000,
    #                          color=(1, 0.2, 0.2), marker='.', s=5)
    # overlay_corners_on_image(img, selected, title=f"ANMS Selected Top {selected.shape[0]}", show_n=None,
    #                          color=(0, 1, 0), marker='x', s=5)

    coords, desc = extract_descriptors(img, selected)

    # 随机挑选一些角点展示对应的 descriptor
    num_show = 10
    idxs = np.random.choice(len(desc), num_show, replace=False)

    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(idxs):
        patch = desc[idx].reshape(8, 8)
        plt.subplot(1, num_show, i + 1)
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
    plt.suptitle("Example 8x8 Normalized Descriptors")
    plt.show()