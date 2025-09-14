import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.signal import convolve2d
from align_image_code import align_images
import cv2
import numpy as np
# First load images

# low sf
# im1 = plt.imread('./DerekPicture.jpg')/255.
im1 = plt.imread('./cat.jpg')
# high sf
# im2 = plt.imread('./nutmeg.jpg')/255
im2 = plt.imread('./dog.jpg')

# Next align images (this code is provided, but may be improved)
im2_aligned, im1_aligned = align_images(im2, im1)
im1_gray = cv2.cvtColor((im1_aligned * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
im2_gray = cv2.cvtColor((im2_aligned * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

im1_gray = im1_gray.astype(np.float32) / 255.0
im2_gray = im2_gray.astype(np.float32) / 255.0

def save_fft(img, filename, title="FFT", cmap="gray"):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))  # log 压缩
    plt.figure()
    plt.imshow(magnitude, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
def hybrid_image(img1, img2, sigma1, sigma2):
    g1 = cv2.getGaussianKernel(ksize=15, sigma=sigma1)
    G1 = g1 @ g1.T
    G1 /= G1.sum()
    low_f = convolve2d(img1, G1, mode="same", boundary="symm")

    g2 = cv2.getGaussianKernel(ksize=5, sigma=sigma2)
    G2 = g2 @ g2.T
    G2 /= G2.sum()
    low_f2 = convolve2d(img2, G2, mode="same", boundary="symm")
    high_f = img2 - low_f2

    result = low_f + high_f * 7.5
    result = np.clip(result, 0, 1)

    return (result * 255).astype(np.uint8), (low_f * 255).astype(np.uint8), (high_f * 255).astype(np.uint8)

sigma1 = 5
sigma2 = 0.5
hybrid, low_f, high_f = hybrid_image(im1_gray, im2_gray, sigma1, sigma2)

cv2.imwrite("hybrid_animal.jpg", hybrid)
cv2.imwrite("low_animal.jpg", low_f)
cv2.imwrite("high_animal.jpg", high_f)

plt.figure(); plt.imshow(hybrid, cmap='gray'); plt.title("Hybrid Image"); plt.axis("off")

save_fft(im1_gray, "fft_input1.jpg", "FFT of input1 (low-pass source)")
save_fft(im2_gray, "fft_input2.jpg", "FFT of input2 (high-pass source)")
save_fft(low_f, "fft_low_pass.jpg", "FFT of low-pass image")
save_fft(high_f, "fft_high_pass.jpg", "FFT of high-pass image")
save_fft(hybrid/255.0, "fft_hybrid.jpg", "FFT of hybrid image")

plt.show()

def pyramids(img, N=5, prefix="pyramid"):
    gaussian_pyr = [img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)]

    # ---- Gaussian Pyramid ----
    for i in range(1, N):
        gaussian_pyr.append(cv2.pyrDown(gaussian_pyr[i - 1]))

    # 保存 Gaussian Pyramid
    gauss_vis = []
    for i, g in enumerate(gaussian_pyr):
        g_show = (g * 255).astype(np.uint8)
        cv2.imwrite(f"{prefix}_gaussian_{i}.jpg", g_show)
        gauss_vis.append(cv2.cvtColor(g_show, cv2.COLOR_GRAY2BGR) if g_show.ndim == 2 else g_show)

    # ---- Laplacian Pyramid ----
    laplacian_pyr = []
    for i in range(N - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        lap = cv2.subtract(gaussian_pyr[i], upsampled)
        laplacian_pyr.append(lap)

    laplacian_pyr.append(gaussian_pyr[-1])

    lap_vis = []
    for i, l in enumerate(laplacian_pyr):
        l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        l_show = l_norm.astype(np.uint8)
        cv2.imwrite(f"{prefix}_laplacian_{i}.jpg", l_show)
        lap_vis.append(cv2.cvtColor(l_show, cv2.COLOR_GRAY2BGR) if l_show.ndim == 2 else l_show)

    gauss_concat = np.hstack([cv2.resize(g, (gauss_vis[0].shape[1], gauss_vis[0].shape[0])) for g in gauss_vis])
    lap_concat = np.hstack([cv2.resize(l, (lap_vis[0].shape[1], lap_vis[0].shape[0])) for l in lap_vis])

    cv2.imwrite(f"{prefix}_gaussian_all.jpg", gauss_concat)
    cv2.imwrite(f"{prefix}_laplacian_all.jpg", lap_concat)

    # print(f"✅ {N} 层 Gaussian & Laplacian Pyramids 已保存")
    return gaussian_pyr, laplacian_pyr

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
pyramids(hybrid, N)
