# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
import os

from skimage.filters import sobel

def edge_ncc_score(img1, img2):
    """基于边缘的 NCC，相比直接像素更鲁棒"""
    e1 = sobel(img1)
    e2 = sobel(img2)
    v1 = e1.flatten() - np.mean(e1)
    v2 = e2.flatten() - np.mean(e2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return -1
    return np.dot(v1, v2) / (norm1 * norm2)

def ncc_score(img1, img2):
    """计算两张图的 NCC 相似度"""
    v1 = img1.flatten()
    v2 = img2.flatten()
    v1 = v1 - np.mean(v1)
    v2 = v2 - np.mean(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return -1
    return np.dot(v1, v2) / (norm1 * norm2)


def crop(img, ratio=0.3):
    """裁剪掉边缘，避免 wrap-around 干扰"""
    h, w = img.shape
    h_margin = int(h * ratio)
    w_margin = int(w * ratio)
    return img[h_margin:h-h_margin, w_margin:w-w_margin]


def align(img, ref, search_range=15):
    """
    在给定 search_range 内 brute-force 搜索最佳平移
    返回对齐后的图像和位移
    """
    best_score = -1
    # best_score = float('inf')  # SSD 最小化
    best_shift = (0, 0)

    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            shifted = np.roll(img, shift=(dx, dy), axis=(0, 1))
            score = ncc_score(crop(ref), crop(shifted))
            # score = np.sum((ref - shifted) ** 2)
            if score > best_score:
                best_score = score
                best_shift = (dx, dy)

    best_img = np.roll(img, shift=best_shift, axis=(0, 1))
    return best_img, best_shift

def pyramid_align(img, ref, max_levels=10, search_range=2, use_edge=False):
    """
    使用图像金字塔 + NCC 的多层对齐
    """
    # === 构建金字塔（降采样） ===
    pyramid_img = [img]
    pyramid_ref = [ref]
    for _ in range(1, max_levels):
        # 每次缩小一半，直到最小 ~64 像素
        if min(pyramid_img[-1].shape) < 64:
            break
        pyramid_img.append(rescale(pyramid_img[-1], 0.5, anti_aliasing=True, channel_axis=None))
        pyramid_ref.append(rescale(pyramid_ref[-1], 0.5, anti_aliasing=True, channel_axis=None))

    levels = len(pyramid_img)

    # 初始位移 (在最小层开始)
    shift = (0, 0)

    # === 从粗到细逐层 refine ===
    for level in reversed(range(levels)):
        # 将前一层位移放大到当前层
        shift = (shift[0] * 2, shift[1] * 2)

        # 在当前层图像应用已有位移
        shifted_img = np.roll(pyramid_img[level], shift=shift, axis=(0, 1))

        # 在局部小范围内搜索最佳 NCC
        best_score = -1
        best_shift = (0, 0)
        for dx in range(-search_range, search_range+1):
            for dy in range(-search_range, search_range+1):
                candidate = np.roll(shifted_img, shift=(dx, dy), axis=(0, 1))
                if use_edge:
                    score = edge_ncc_score(crop(pyramid_ref[level]), crop(candidate))
                else:
                    score = ncc_score(crop(pyramid_ref[level]), crop(candidate))
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

        # 更新全局位移
        shift = (shift[0] + best_shift[0], shift[1] + best_shift[1])

        print(f"Level {level}: dx={shift[0]}, dy={shift[1]}, NCC={best_score:.4f}")

    # === 在原图应用最终位移 ===
    final_aligned = np.roll(img, shift=shift, axis=(0, 1))
    print(f"[Pyramid] Final shift: dx={shift[0]}, dy={shift[1]}")
    return final_aligned

names = [
        'cathedral.jpg',
        'church.tif',
        'emir.tif',
        'harvesters.tif',
        'icon.tif',
        'italil.tif',
        'lastochikino.tif',
        'lugano.tif',
        'melons.tif',
        'monastery.jpg',
        'self_portrait.tif',
        'siren.tif',
        'three_generations.tif',
        'tobolsk.jpg',
        'master-pnp-prok-00000-00087a.tif',
        'master-pnp-prok-00100-00178a.tif',
        'master-pnp-prok-00200-00220a.tif'
         ]

for imname in names:
    # name of the input file
    # imname = 'emir.tif'
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    # height = np.floor(im.shape[0] / 3.0).astype(np.int)
    height = int(np.floor(im.shape[0] / 3.0))

    # separate color channels
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    # 使用金字塔对齐
    ag = pyramid_align(g, b, use_edge=True)
    # R 通道用 Edge-NCC（专门针对 Emir）
    # if "emir" in imname.lower():
    #     ar = pyramid_align(r, b, use_edge=True)
    # else:
    #     ar = pyramid_align(r, b)
    ar = pyramid_align(r, b, use_edge=True)
    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    # 输出路径
    out_dir = 'out_path'
    # 保留原始文件名
    fname = os.path.join(out_dir, os.path.splitext(os.path.basename(imname))[0] + '_output.jpg')

    # skio.imsave(fname, im_out)

    # 转成 [0,255] 的 uint8
    im_out_uint8 = sk.img_as_ubyte(im_out)

    skio.imsave(fname, im_out_uint8)

    # display the image
    skio.imshow(im_out)
    skio.show()

