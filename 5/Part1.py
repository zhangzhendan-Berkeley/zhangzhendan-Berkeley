from PIL import Image
import mediapy as media
from pprint import pprint
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

# For downloading web images
import requests
from io import BytesIO
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt
import os
import numpy as np

device = 'cuda'

def seed_everything(seed):
  torch.cuda.manual_seed(seed)
  torch.manual_seed(seed)

# Load DeepFloyd IF stage I
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-L-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
)
stage_1.to(device)

# Load DeepFloyd IF stage II
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
)
stage_2.to(device)

YOUR_SEED = 314159
seed_everything(YOUR_SEED)
# Get scheduler parameters
alphas_cumprod = stage_1.scheduler.alphas_cumprod
print(f"We have in total {alphas_cumprod.shape[0]} noise coefficients")

# Get test image
# test_im = Image.open('campanile.jpg')

# For stage 1: Resize to (64, 64), convert to tensor, rescale to [-1, 1], and
# add a batch dimension. The result is a (1, 3, 64, 64) tensor
test_im = Image.open('campanile.jpg').resize((64, 64))
test_im = TF.to_tensor(test_im)
test_im = 2 * test_im - 1
test_im = test_im[None]

# Show test image



# Tensor is [-1,1], convert to [0,1] for display
img = (test_im[0].permute(1,2,0).cpu().numpy() / 2) + 0.5
# print('Test image:')
# plt.imshow(img)
# plt.axis("off")
# plt.show()

def forward(im, t):
    """
    Args:
      im : torch tensor of size (1, 3, 64, 64) representing x0 (clean image)
      t : integer timestep

    Returns:
      im_noisy : torch tensor of size (1, 3, 64, 64) representing xt (noisy image)
    """
    with torch.no_grad():
        # Get alpha_bar_t
        alpha_bar = alphas_cumprod[t]   # scalar tensor

        # Sample noise eps ~ N(0, I)
        eps = torch.randn_like(im)

        # q(xt | x0) = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps
        im_noisy = (alpha_bar.sqrt() * im) + ((1 - alpha_bar).sqrt() * eps)

    return im_noisy


# Show the test image at noise level [250, 500, 750]

timesteps = [250, 500, 750]


# os.makedirs("1_1", exist_ok=True)
#
# for t in timesteps:
#     im_noisy = forward(test_im, t)  # your forward() function
#
#     # convert from [-1,1] → [0,1]
#     img = (im_noisy[0].permute(1, 2, 0).cpu().numpy() / 2) + 0.5
#
#     plt.figure(figsize=(4, 4))
#     plt.imshow(img)
#     plt.title(f"Noise level t = {t}")
#     plt.axis("off")
#
#     # save before showing
#     save_path = f"1_1/noise_t_{t}.png"
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     print(f"Saved: {save_path}")
#
#     plt.show()



# os.makedirs("1_2", exist_ok=True)
#
# for t in timesteps:
#     # Re-generate noisy image
#     im_noisy = forward(test_im, t)
#
#     # Classical gaussian blur denoising
#     # You can tweak kernel_size or sigma if you want
#     im_denoised = gaussian_blur(im_noisy, kernel_size=5, sigma=1)
#
#     # Convert both to [0, 1] for display
#     noisy_np = (im_noisy[0].permute(1, 2, 0).cpu().numpy() / 2) + 0.5
#     denoised_np = (im_denoised[0].permute(1, 2, 0).cpu().numpy() / 2) + 0.5
#
#     # Plot side by side
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#     axs[0].imshow(noisy_np)
#     axs[0].set_title(f"Noisy Image (t={t})")
#     axs[0].axis("off")
#
#     axs[1].imshow(denoised_np)
#     axs[1].set_title("Gaussian Denoised")
#     axs[1].axis("off")
#
#     # Save figure
#     save_path = f"1_2/gaussian_denoise_t_{t}.png"
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     print(f"Saved: {save_path}")
#
#     plt.show()

def denoise_one_step(x0_clean, t, prompt_embeds):
    """
    x0_clean: 原始干净图像 (1,3,64,64)，在 CPU 或 GPU 都可以
    t: int, timestep
    prompt_embeds: 对应 prompt 的 embedding, shape (1, 77, 4096)
    """
    with torch.no_grad():
        # 1. 用 forward 加噪得到 x_t
        x0_clean = x0_clean.to(device)
        x_t = forward(x0_clean, t)          # (1,3,64,64)，仍在 device 上（如果你在 forward 里做了 .to(device) 就不用再转）

        # 2. 准备 timestep tensor
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)

        # 3. 准备输入给 UNet：half 精度
        x_t_in = x_t.half()
        prompt_embeds_in = prompt_embeds.to(device).half()

        # 4. 通过 UNet 预测噪声
        # 返回 (eps_pred, var_pred) 但我们只用前者
        eps_and_var = stage_1.unet(
            x_t_in,
            t_tensor,
            encoder_hidden_states=prompt_embeds_in,
            return_dict=False
        )
        eps_pred = eps_and_var[0][:, :3]  # 取前 3 个通道作为噪声估计，shape (1,3,64,64)

        # 5. 取出 alpha_bar_t
        alpha_bar_t = alphas_cumprod[t].to(device)           # scalar
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

        # 6. 使用公式： x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        x0_hat = (x_t_in - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

        # 输出用 float32 更方便后面处理
        x0_hat = x0_hat.float()

    return x_t.float(), x0_hat
#
# os.makedirs("1_3", exist_ok=True)

# Loads your own .pth file here
prompt_embeds_dict = torch.load('prompt_embeds_dict.pth')

# Please use the null prompt embedding
prompt_embeds = prompt_embeds_dict["a high quality photo"].half().cuda()
#
# with torch.no_grad():
#   for t in [250, 500, 750]:
#     # Get alpha bar
#     alpha_cumprod = alphas_cumprod[t]            # scalar
#     sqrt_alpha = alpha_cumprod.sqrt()
#     sqrt_one_minus_alpha = (1 - alpha_cumprod).sqrt()
#
#     # Run forward process (add noise)
#     # ===== your code here! =====
#
#     eps = torch.randn_like(test_im)              # noise
#     im_noisy = sqrt_alpha * test_im + sqrt_one_minus_alpha * eps
#
#     # ==== end of code ====
#
#     # Estimate noise in noisy image
#     noise_est = stage_1.unet(
#         im_noisy.half().cuda(),
#         t,
#         encoder_hidden_states=prompt_embeds,
#         return_dict=False
#     )[0]
#
#     # Take only first 3 channels, and move result to cpu
#     noise_est = noise_est[:, :3].cpu()
#
#     # Remove the noise (estimate x0 using Equation 2)
#     # ===== your code here! =====
#
#     im_recon = (im_noisy.cpu() - sqrt_one_minus_alpha * noise_est) / sqrt_alpha
#
#     # ==== end of code ====
#
#     # Convert to numpy for display
#     orig_img = (test_im[0].permute(1,2,0).numpy() / 2) + 0.5
#     noisy_img = (im_noisy[0].permute(1,2,0).cpu().numpy() / 2) + 0.5
#     recon_img = (im_recon[0].permute(1,2,0).numpy() / 2) + 0.5
#
#     # Visualize
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,3,1)
#     plt.imshow(orig_img)
#     plt.title("Original")
#     plt.axis("off")
#
#     plt.subplot(1,3,2)
#     plt.imshow(noisy_img)
#     plt.title(f"Noisy (t={t})")
#     plt.axis("off")
#
#     plt.subplot(1,3,3)
#     plt.imshow(recon_img)
#     plt.title("Reconstructed x0")
#     plt.axis("off")
#
#     plt.show()

# create `strided_timesteps`, a list of timesteps, from 990 to 0 in steps of 30
strided_timesteps = list(range(990, -1, -30))
print(strided_timesteps)
# e.g. [990, 960, 930, ..., 30, 0]

stage_1.scheduler.set_timesteps(timesteps=strided_timesteps)

def add_variance(predicted_variance, t_index, image):
    """
    Args:
      predicted_variance: (1,3,64,64) tensor, 来自 UNet 输出后 3 个通道
      t_index: int，当前的时间步（strided_timesteps[i] 的那个 int）
      image: (1,3,64,64) 当前图像

    Returns:
      (1,3,64,64) tensor，加上方差噪声后的图像
    """
    # 让 scheduler 根据 predicted_variance 的 device/dtype 自己搬运
    variance = stage_1.scheduler._get_variance(t_index, predicted_variance=predicted_variance)
    # variance 跟 predicted_variance 在同一个 device / dtype

    variance_noise = torch.randn_like(image)
    variance = torch.exp(0.5 * variance) * variance_noise
    return image + variance


def iterative_denoise(im_noisy, i_start, prompt_embeds, timesteps, display=True):
    image = im_noisy

    with torch.no_grad():
        for i in range(i_start, len(timesteps) - 1):

            # Get timesteps
            t = timesteps[i]
            t_prev = timesteps[i+1]

            # ᾱ
            alpha_bar_t = alphas_cumprod[t].to(device)
            alpha_bar_prev = alphas_cumprod[t_prev].to(device)

            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t  = 1 - alpha_t

            # UNet forward
            model_output = stage_1.unet(
                image,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]

            noise_est, predicted_variance = torch.split(model_output, 3, dim=1)

            # x0_hat
            x0_hat = (image - torch.sqrt(1 - alpha_bar_t) * noise_est) / torch.sqrt(alpha_bar_t)

            # Equation 3
            term1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat
            term2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t) * image
            x_prev = term1 + term2

            x_prev = add_variance(predicted_variance, t, x_prev)
            image = x_prev

            # === Save every 5 steps ===
            if i % 5 == 0:
                img_np = image[0].detach().cpu().permute(1,2,0).float().numpy()
                img_np = img_np / 2 + 0.5
                img_np = np.clip(img_np, 0, 1)

                # save_path = f"1_4/iter_step_{i}_t_{t}.png"
                # plt.imsave(save_path, img_np)
                # print("Saved:", save_path)

                if display:
                    plt.imshow(img_np)
                    plt.title(f"step {i}, t={t}")
                    plt.axis("off")
                    plt.show()

        # Final clean
        clean = image[0].detach().cpu().permute(1,2,0).float().numpy()
        clean = clean / 2 + 0.5
        clean = np.clip(clean, 0, 1)

        # Save final clean
        # final_path = "1_4/iter_final.png"
        # plt.imsave(final_path, clean)
        # print("Saved:", final_path)

        return clean


# # Add noise
# i_start = 10
# t = strided_timesteps[i_start]
# im_noisy = forward(test_im, t).half().to(device)
#
# # Denoise
# clean = iterative_denoise(im_noisy,
#                           i_start=i_start,
#                           prompt_embeds=prompt_embeds,
#                           timesteps=strided_timesteps)
#
# # Compute the one step estimate of the clean image (from part 1.3)
# with torch.no_grad():
#     alpha_bar = alphas_cumprod[t].to(device)
#     sqrt_alpha = torch.sqrt(alpha_bar)
#     sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)
#
#     model_output = stage_1.unet(
#         im_noisy,
#         t,
#         encoder_hidden_states=prompt_embeds,
#         return_dict=False
#     )[0]
#     noise_est = model_output[:, :3]
#
#     clean_one_step = (im_noisy - sqrt_one_minus_alpha * noise_est) / sqrt_alpha
#     clean_one_step = clean_one_step[0].permute(1,2,0).cpu().numpy() / 2 + 0.5
#
# blur_img = gaussian_blur(im_noisy.cpu(), kernel_size=5, sigma=2)
# blur_filtered = blur_img[0].permute(1,2,0).numpy() / 2 + 0.5
#
# os.makedirs("1_4", exist_ok=True)
#
# # Original
# orig_np = test_im[0].detach().cpu().permute(1,2,0).numpy()
# orig_np = orig_np / 2 + 0.5
# orig_np = np.clip(orig_np, 0, 1).astype(np.float32)
#
# # Iterative denoising result
# iter_np = clean   # clean is already numpy (64,64,3)
# iter_np = np.clip(iter_np.astype(np.float32), 0, 1)
#
# # One-step denoising
# one_np = np.clip(clean_one_step.astype(np.float32), 0, 1)
#
# # Gaussian blur
# gb_np = np.clip(blur_filtered.astype(np.float32), 0, 1)
#
# # ---- Display & Save ----
# fig, axs = plt.subplots(1,4, figsize=(16,4))
# axs[0].imshow(orig_np); axs[0].set_title("Original"); axs[0].axis("off")
# axs[1].imshow(iter_np); axs[1].set_title("Iterative"); axs[1].axis("off")
# axs[2].imshow(one_np);  axs[2].set_title("One-step"); axs[2].axis("off")
# axs[3].imshow(gb_np);   axs[3].set_title("Gaussian blur"); axs[3].axis("off")
#
# save_path = "1_4/comparison.png"
# plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
# plt.show()
#
# print(f"Saved comparison figure to {save_path}")

# # Please use this text prompt embedding
# prompt_embeds = prompt_embeds_dict["a high quality photo"]
# prompt_embeds = prompt_embeds.half().to(device)
# os.makedirs("1_5", exist_ok=True)
#
# sampled_images = []
#
# for idx in range(5):
#     # 1. Start from pure noise
#     noise = torch.randn((1, 3, 64, 64)).half().to(device)
#
#     # 2. Run full denoising from t=990 → 0
#     clean = iterative_denoise(
#         im_noisy=noise,
#         i_start=0,                      # start from the very first timestep
#         prompt_embeds=prompt_embeds,
#         timesteps=strided_timesteps,
#         display=False                   # 不需要显示中间步骤
#     )
#
#     sampled_images.append(clean)
#
#     # 3. Save image
#     save_path = f"1_5/sample_{idx+1}.png"
#     plt.imsave(save_path, clean)
#     print("Saved:", save_path)
#
# # Show all 5 images
# fig, axs = plt.subplots(1, 5, figsize=(20, 4))
# for i in range(5):
#     axs[i].imshow(sampled_images[i])
#     axs[i].set_title(f"sample {i+1}")
#     axs[i].axis("off")
#
# plt.show()

def iterative_denoise_cfg(
        im_noisy,
        i_start,
        prompt_embeds,
        uncond_prompt_embeds,
        timesteps,
        scale=7,
        display=True
):
    image = im_noisy

    with torch.no_grad():
        for i in range(i_start, len(timesteps) - 1):

            # t and t'
            t = timesteps[i]
            t_prev = timesteps[i + 1]

            # ------- ᾱ_t and ᾱ_t' -------
            alpha_bar_t = alphas_cumprod[t].to(device)
            alpha_bar_prev = alphas_cumprod[t_prev].to(device)

            # ------- α_t and β_t -------
            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = 1 - alpha_t

            # ------- conditional noise -------
            model_output = stage_1.unet(
                image,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]
            noise_est, predicted_variance = torch.split(model_output, 3, dim=1)

            # ------- unconditional noise -------
            uncond_output = stage_1.unet(
                image,
                t,
                encoder_hidden_states=uncond_prompt_embeds,
                return_dict=False
            )[0]
            uncond_noise_est, _ = torch.split(uncond_output, 3, dim=1)

            # ---------- CFG noise (Equation 4) ----------
            #  ε = ε_u + γ(ε_c - ε_u)
            guided_noise = uncond_noise_est + scale * (noise_est - uncond_noise_est)

            # ---------- Estimate x0 ----------
            x0_hat = (image - torch.sqrt(1 - alpha_bar_t) * guided_noise) / torch.sqrt(alpha_bar_t)

            # ---------- Equation 3 ----------
            term1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat
            term2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t) * image

            x_prev = term1 + term2

            # add variance using conditional predicted_variance
            x_prev = add_variance(predicted_variance, t, x_prev)

            image = x_prev

            # ---------- Display every 5 steps ----------
            if display and (i % 5 == 0):
                img_np = image[0].detach().cpu().permute(1, 2, 0).float().numpy()
                img_np = img_np / 2 + 0.5
                img_np = np.clip(img_np, 0, 1)
                plt.imshow(img_np)
                plt.title(f"CFG step {i}, t={t}")
                plt.axis("off")
                plt.show()

        # final clean image
        clean = image[0].detach().cpu().permute(1, 2, 0).float().numpy()
        clean = clean / 2 + 0.5
        clean = np.clip(clean, 0, 1)

        return clean

# os.makedirs("1_6", exist_ok=True)
#
# samples = []
# prompt_embeds = prompt_embeds_dict["a high quality photo"].half().to(device)
# uncond_prompt_embeds = prompt_embeds_dict[""].half().to(device)
#
# for idx in range(5):
#     noise = torch.randn((1,3,64,64)).half().to(device)
#
#     clean = iterative_denoise_cfg(
#         im_noisy=noise,
#         i_start=0,
#         prompt_embeds=prompt_embeds,
#         uncond_prompt_embeds=uncond_prompt_embeds,
#         timesteps=strided_timesteps,
#         scale=7,
#         display=False
#     )
#
#     samples.append(clean)
#     plt.imsave(f"1_6/cfg_sample_{idx+1}.png", clean)
#     print("Saved:", f"1_6/cfg_sample_{idx+1}.png")
#
# # display
# fig, axs = plt.subplots(1, 5, figsize=(20,4))
# for i in range(5):
#     axs[i].imshow(samples[i])
#     axs[i].set_title(f"sample {i+1}")
#     axs[i].axis("off")
# plt.show()

prompt_embeds = prompt_embeds_dict["a high quality photo"].half().to(device)
uncond_prompt_embeds = prompt_embeds_dict[""].half().to(device)

# os.makedirs("1_7", exist_ok=True)

# These are the starting steps requested
# i_starts = [1, 3, 5, 7, 10, 20]

# results = []

# for i_start in i_starts:
#     t = strided_timesteps[i_start]
#
#     # 1. Add noise to original Campanile
#     im_noisy = forward(test_im, t).half().to(device)
#
#     # 2. Run CFG denoising starting from i_start
#     clean = iterative_denoise_cfg(
#         im_noisy=im_noisy,
#         i_start=i_start,
#         prompt_embeds=prompt_embeds,
#         uncond_prompt_embeds=uncond_prompt_embeds,
#         timesteps=strided_timesteps,
#         scale=7,
#         display=False
#     )
#
#     results.append(clean)
#
#     # save
#     save_path = f"1_7/campanile_edit_i_start_{i_start}.png"
#     plt.imsave(save_path, clean)
#     print("Saved:", save_path)
#
# # visualize
# fig, axs = plt.subplots(1, len(i_starts), figsize=(20,4))
# for j, start in enumerate(i_starts):
#     axs[j].imshow(results[j])
#     axs[j].set_title(f"i_start={start}")
#     axs[j].axis("off")
# plt.show()

def load_img_64(path):
    im = Image.open(path).resize((64, 64))
    im = TF.to_tensor(im)
    im = 2 * im - 1
    return im[None]

# my_imgs = [
#     load_img_64("my1.jpg"),
#     load_img_64("my2.jpg")
# ]
#
# for img_idx, my_im in enumerate(my_imgs):
#     print(f"Processing my image {img_idx+1}")
#
#     for i_start in i_starts:
#         t = strided_timesteps[i_start]
#
#         im_noisy = forward(my_im, t).half().to(device)
#
#         clean = iterative_denoise_cfg(
#             im_noisy=im_noisy,
#             i_start=i_start,
#             prompt_embeds=prompt_embeds,
#             uncond_prompt_embeds=uncond_prompt_embeds,
#             timesteps=strided_timesteps,
#             scale=7,
#             display=False
#         )
#
#         # save
#         save_path = f"1_7/myimg{img_idx+1}_i_start_{i_start}.png"
#         plt.imsave(save_path, clean)
#         print("Saved:", save_path)

def process_pil_im(img):
    """
    Converts PIL image → tensor (1,3,64,64) in [-1,1]
    """
    img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    img = transform(img)[None]

    # local preview
    plt.imshow((img[0].permute(1,2,0).numpy() / 2 + 0.5).clip(0,1))
    plt.title("Processed input")
    plt.axis("off")
    plt.show()

    return img

os.makedirs("1_7_1", exist_ok=True)

i_starts = [1, 3, 5, 7, 10, 20]

def sdedit_image(im_tensor, name_prefix):
    """
    Apply SDEdit using iterative_denoise_cfg over 6 noise levels.
    Save outputs into 1_8 folder.
    """

    for i_start in i_starts:
        print(f"Running SDEdit: {name_prefix}, i_start={i_start}")

        t = strided_timesteps[i_start]

        # 1. Add noise
        noisy = forward(im_tensor, t).half().to(device)

        # 2. CFG denoise
        clean = iterative_denoise_cfg(
            im_noisy=noisy,
            i_start=i_start,
            prompt_embeds=prompt_embeds,
            uncond_prompt_embeds=uncond_prompt_embeds,
            timesteps=strided_timesteps,
            scale=7,
            display=False
        )

        clean_np = clean / 2 + 0.5
        clean_np = clean_np.clip(0,1)

        # 3. Save image
        path = f"1_7_1/{name_prefix}_i_start_{i_start}.png"
        plt.imsave(path, clean_np)
        print("Saved:", path)

# anime = process_pil_im(Image.open("anime.png"))
# sdedit_image(anime, "anime")
#
# my1 = process_pil_im(Image.open("mydraw1.jpg"))
# my2 = process_pil_im(Image.open("mydraw2.jpg"))
#
# sdedit_image(my1, "mydraw1")
# sdedit_image(my2, "mydraw2")

def inpaint(original_image, mask, prompt_embeds, uncond_prompt_embeds, timesteps, scale=7, display=True):
    """
    original_image: (1,3,64,64) in [-1,1]
    mask: same shape, 1 = to generate, 0 = keep original content
    """
    # Start from pure noise
    image = torch.randn_like(original_image).to(device).half()

    with torch.no_grad():
        for i in range(len(timesteps) - 1):

            t = timesteps[i]
            t_prev = timesteps[i + 1]

            # ᾱ values
            alpha_bar_t = alphas_cumprod[t].to(device)
            alpha_bar_prev = alphas_cumprod[t_prev].to(device)

            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t = 1 - alpha_t

            # ---------- conditional noise ----------
            model_output = stage_1.unet(
                image,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]
            noise_est, predicted_variance = torch.split(model_output, 3, dim=1)

            # ---------- unconditional ----------
            uncond_output = stage_1.unet(
                image,
                t,
                encoder_hidden_states=uncond_prompt_embeds,
                return_dict=False
            )[0]
            uncond_noise_est, _ = torch.split(uncond_output, 3, dim=1)

            # ---------- CFG ----------
            guided_noise = uncond_noise_est + scale * (noise_est - uncond_noise_est)

            # ---------- reconstruct x0 ----------
            x0_hat = (image - torch.sqrt(1 - alpha_bar_t) * guided_noise) / torch.sqrt(alpha_bar_t)

            # ---------- DDPM update ----------
            term1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat
            term2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t) * image
            x_prev = term1 + term2

            # add variance
            x_prev = add_variance(predicted_variance, t, x_prev)

            # ============================================================
            # ✅ *关键一步：inpainting 覆盖 mask==0 的区域*
            # ============================================================
            # correct xt for original image outside the mask
            x_orig_t = forward(original_image, t).to(device).half()
            x_prev = (mask * x_prev + (1 - mask) * x_orig_t).half()

            # ============================================================

            image = x_prev

            if display and (i % 10 == 0):
                img_np = image[0].detach().cpu().permute(1,2,0).float().numpy()
                img_np = img_np / 2 + 0.5
                img_np = np.clip(img_np, 0, 1)
                plt.imshow(img_np)
                plt.title(f"Inpainting step {i}, t={t}")
                plt.axis("off")
                plt.show()

        # Final clean image
        clean = image[0].detach().cpu().permute(1,2,0).float().numpy()
        clean = clean / 2 + 0.5
        clean = np.clip(clean, 0, 1)

        return clean

# # -------------------------
# # Make output directory
# # -------------------------
# os.makedirs("1_7_2", exist_ok=True)
#
# # -------------------------
# # 1. Create mask
# # -------------------------
# mask = torch.zeros_like(test_im)
# mask[:, :, 2:20, 24:42] = 1.0
# mask = mask.to(device)
#
# # -------------------------
# # 2. Run inpainting
# # -------------------------
# result = inpaint(
#     original_image=test_im.to(device),
#     mask=mask,
#     prompt_embeds=prompt_embeds,
#     uncond_prompt_embeds=uncond_prompt_embeds,
#     timesteps=strided_timesteps,
#     scale=7,
#     display=False
# )
#
# # -------------------------
# # 3. Convert numpy → [0,1]
# # -------------------------
# result_np = result
# orig_np = (test_im[0].permute(1,2,0).cpu().numpy() / 2 + 0.5).clip(0,1)
# mask_np = mask[0].permute(1,2,0).detach().cpu().numpy()
#
# # FIXED VERSION → test_im.to(device)
# masked_input_np = ((test_im.to(device) * mask))[0].permute(1,2,0).detach().cpu().numpy() / 2 + 0.5
# masked_input_np = masked_input_np.clip(0,1)
#
# # -------------------------
# # 4. Visualization
# # -------------------------
# fig, axs = plt.subplots(1, 4, figsize=(16,4))
#
# axs[0].imshow(orig_np)
# axs[0].set_title("Original")
# axs[0].axis("off")
#
# axs[1].imshow(mask_np)
# axs[1].set_title("Mask (white = edit)")
# axs[1].axis("off")
#
# axs[2].imshow(masked_input_np)
# axs[2].set_title("Original × Mask")
# axs[2].axis("off")
#
# axs[3].imshow(result_np)
# axs[3].set_title("Inpainted Result")
# axs[3].axis("off")
#
# plt.tight_layout()
# plt.show()
#
# # -------------------------
# # 5. Save everything to 1_7_2/
# # -------------------------
# plt.imsave("1_7_2/campanile_inpaint_result.png", result_np)
# plt.imsave("1_7_2/campanile_mask.png", mask_np)
# plt.imsave("1_7_2/campanile_masked_input.png", masked_input_np)
# plt.imsave("1_7_2/campanile_original.png", orig_np)
#
# print("Saved to folder: 1_7_2/")

def run_inpaint_for_my_image(img_path, mask_coords, outname):
    """
    img_path: path to your JPG/PNG
    mask_coords: (y1, y2, x1, x2)
    outname: folder name under 1_7_2/
    """

    # -------------------------
    # load & preprocess image
    # -------------------------
    img = Image.open(img_path).convert("RGB").resize((64, 64))
    img_t = TF.to_tensor(img)
    img_t = 2 * img_t - 1
    img_t = img_t[None]          # still on CPU here

    # -------------------------
    # Prepare directory
    # -------------------------
    savedir = f"1_7_2/{outname}"
    os.makedirs(savedir, exist_ok=True)

    # -------------------------
    # create mask (GPU)
    # -------------------------
    mask = torch.zeros_like(img_t)
    y1, y2, x1, x2 = mask_coords
    mask[:, :, y1:y2, x1:x2] = 1
    mask = mask.to(device)

    # -------------------------
    # Run inpaint (img to GPU)
    # -------------------------
    result = inpaint(
        original_image=img_t.to(device),
        mask=mask,
        prompt_embeds=prompt_embeds,
        uncond_prompt_embeds=uncond_prompt_embeds,
        timesteps=strided_timesteps,
        scale=7,
        display=False
    )

    # -------------------------
    # convert numpy
    # -------------------------
    orig_np = (img_t[0].permute(1,2,0).numpy() / 2 + 0.5).clip(0,1)
    mask_np = mask[0].permute(1,2,0).detach().cpu().numpy()

    # FIXED: move img to GPU before multiply
    masked_input_np = ((img_t.to(device) * mask))[0].permute(1,2,0).detach().cpu().numpy() / 2 + 0.5
    masked_input_np = masked_input_np.clip(0,1)

    result_np = result   # already numpy [0,1]

    # -------------------------
    # Visualization
    # -------------------------
    fig, axs = plt.subplots(1, 4, figsize=(16,4))

    axs[0].imshow(orig_np); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(mask_np); axs[1].set_title("Mask"); axs[1].axis("off")
    axs[2].imshow(masked_input_np); axs[2].set_title("Original×Mask"); axs[2].axis("off")
    axs[3].imshow(result_np); axs[3].set_title("Inpainted"); axs[3].axis("off")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # save all
    # -------------------------
    plt.imsave(f"{savedir}/original.png", orig_np)
    plt.imsave(f"{savedir}/mask.png", mask_np)
    plt.imsave(f"{savedir}/masked_input.png", masked_input_np)
    plt.imsave(f"{savedir}/inpaint_result.png", result_np)

    print(f"[Saved all results to] {savedir}")

# run_inpaint_for_my_image("mydraw1.jpg", (10,40,20,50), "mydraw1")
# run_inpaint_for_my_image("mydraw2.jpg", (12,45,15,48), "mydraw2")



# ============================================
# 1. 选择文本 prompt
# ============================================
text_prompt = "a rocket ship"   # ← 你可以换成任何 prompt

prompt_embeds = prompt_embeds_dict[text_prompt].half().to(device)
uncond_prompt_embeds = prompt_embeds_dict[""].half().to(device)

i_starts = [1, 3, 5, 7, 10, 20]

# ============================================
# 2. 确保输出目录存在
# ============================================
os.makedirs("1_7_3/campanile", exist_ok=True)
os.makedirs("1_7_3/mydraw1", exist_ok=True)
os.makedirs("1_7_3/mydraw2", exist_ok=True)


# ============================================
# 3. 定义一个通用函数：Text-Guided SDEdit
# ============================================
def sdedit_text_guided(img_t, outdir):
    """
    img_t: (1,3,64,64) preprocessed image
    outdir: where to save results
    """
    results = []

    for i_start in i_starts:
        t = strided_timesteps[i_start]

        # 1. add noise
        noisy = forward(img_t, t).half().to(device)

        # 2. denoise with CFG
        clean = iterative_denoise_cfg(
            im_noisy=noisy,
            i_start=i_start,
            prompt_embeds=prompt_embeds,
            uncond_prompt_embeds=uncond_prompt_embeds,
            timesteps=strided_timesteps,
            scale=7,
            display=False
        )

        results.append(clean)

        # 3. save
        save_path = f"{outdir}/edit_i_start_{i_start}.png"
        plt.imsave(save_path, clean)
        print("Saved:", save_path)

    return results


# ============================================
# 4. Campanile processing
# ============================================
# campanile_results = sdedit_text_guided(test_im.to(device), "1_7_3/campanile")

# ============================================
# 5. Process your own images
# ============================================

def load_image64(path):
    img = Image.open(path).convert("RGB").resize((64, 64))
    img_t = TF.to_tensor(img)
    img_t = 2 * img_t - 1
    return img_t[None].to(device)


mydraw1 = load_image64("mydraw1.jpg")
mydraw2 = load_image64("mydraw2.jpg")

# my1_results = sdedit_text_guided(mydraw1, "1_7_3/mydraw1")
# my2_results = sdedit_text_guided(mydraw2, "1_7_3/mydraw2")

def make_flip_illusion(
    image,
    i_start,
    prompt1_embeds,
    prompt2_embeds,
    uncond_embeds,
    timesteps,
    scale=7,
    display=True
):
    """
    image: (1,3,64,64) noisy starting image (usually pure noise)
    i_start: starting index in strided_timesteps (0 for full generation)
    prompt1_embeds: embedding for "upright" prompt
    prompt2_embeds: embedding for "upside-down" prompt
    uncond_embeds: embedding for unconditional CFG branch
    timesteps: list of t
    """

    x = image

    with torch.no_grad():

        for i in range(i_start, len(timesteps)-1):

            t      = timesteps[i]
            t_prev = timesteps[i+1]

            # ---- α(t) computation (same as before) ----
            alpha_bar_t    = alphas_cumprod[t].to(device)
            alpha_bar_prev = alphas_cumprod[t_prev].to(device)

            alpha_t = alpha_bar_t / alpha_bar_prev
            beta_t  = 1 - alpha_t

            # ==============================
            # 1. NORMAL BRANCH (upright)
            # ==============================
            out1 = stage_1.unet(
                x,
                t,
                encoder_hidden_states=prompt1_embeds,
                return_dict=False
            )[0]

            eps1, var1 = torch.split(out1, 3, dim=1)

            # unconditional branch for CFG
            out1u = stage_1.unet(
                x,
                t,
                encoder_hidden_states=uncond_embeds,
                return_dict=False
            )[0]
            eps1u, _ = torch.split(out1u, 3, dim=1)

            # CFG noise
            eps1 = eps1u + scale * (eps1 - eps1u)


            # ==============================
            # 2. FLIPPED BRANCH (upside-down)
            # ==============================
            x_flip = torch.flip(x, dims=[2])   # flip vertically (H dimension)

            out2 = stage_1.unet(
                x_flip,
                t,
                encoder_hidden_states=prompt2_embeds,
                return_dict=False
            )[0]

            eps2, _ = torch.split(out2, 3, dim=1)

            # unconditional
            out2u = stage_1.unet(
                x_flip,
                t,
                encoder_hidden_states=uncond_embeds,
                return_dict=False
            )[0]
            eps2u, _ = torch.split(out2u, 3, dim=1)

            # CFG
            eps2 = eps2u + scale * (eps2 - eps2u)

            # flip noise back
            eps2 = torch.flip(eps2, dims=[2])


            # ==============================
            # 3. Combine noise estimates
            # ==============================
            eps = 0.5 * (eps1 + eps2)

            # ==============================
            # 4. Estimate x0
            # ==============================
            x0_hat = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # ==============================
            # 5. DDPM sampling step
            # ==============================
            term1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat
            term2 = (torch.sqrt(alpha_t)         * (1 - alpha_bar_prev)) / (1 - alpha_bar_t) * x

            x_prev = term1 + term2

            # add variance (using var1 from upright branch)
            x_prev = add_variance(var1, t, x_prev)

            x = x_prev

            # optional display
            if display and (i % 10 == 0):
                tmp = x[0].detach().cpu().permute(1,2,0).float().numpy()
                tmp = tmp / 2 + 0.5
                tmp = np.clip(tmp, 0, 1)
                plt.imshow(tmp)
                plt.title(f"Illusion step {i}, t={t}")
                plt.axis("off")
                plt.show()

        # final image
        final = x[0].detach().cpu().permute(1,2,0).float().numpy()
        final = final / 2 + 0.5
        final = np.clip(final, 0, 1)

        return final

prompt1 = prompt_embeds_dict["a watercolor painting of a young girl smiling"].half().to(device)
prompt2 = prompt_embeds_dict["a watercolor of a blooming tree with twisted roots"].half().to(device)
uncond = prompt_embeds_dict[""].half().to(device)

# start from pure noise
noise = torch.randn((1,3,64,64)).half().to(device)

illusion = make_flip_illusion(
    image=noise,
    i_start=0,
    prompt1_embeds=prompt1,
    prompt2_embeds=prompt2,
    uncond_embeds=uncond,
    timesteps=strided_timesteps,
    scale=7,
    display=False
)


plt.imshow(illusion)
plt.axis("off")
plt.show()

plt.imsave("illusion.png", illusion)

# def make_hybrids(
#     image,
#     i_start,
#     prompt1_embeds,
#     prompt2_embeds,
#     uncond_embeds,
#     timesteps,
#     scale=7,
#     display=True
# ):
#     """
#     Hybrid images using diffusion model noise factorization.
#     image: (1,3,64,64)
#     """
#     x = image
#
#     with torch.no_grad():
#
#         for i in range(i_start, len(timesteps)-1):
#
#             t      = timesteps[i]
#             t_prev = timesteps[i+1]
#
#             # ---- alpha(t) computation ----
#             alpha_bar_t    = alphas_cumprod[t].to(device)
#             alpha_bar_prev = alphas_cumprod[t_prev].to(device)
#
#             alpha_t = alpha_bar_t / alpha_bar_prev
#             beta_t  = 1 - alpha_t
#
#             # ==============================
#             # 1. ε1 = CFG(xt, prompt1)
#             # ==============================
#             out1 = stage_1.unet(
#                 x,
#                 t,
#                 encoder_hidden_states=prompt1_embeds,
#                 return_dict=False
#             )[0]
#             eps1, var1 = torch.split(out1, 3, dim=1)
#
#             out1u = stage_1.unet(
#                 x,
#                 t,
#                 encoder_hidden_states=uncond_embeds,
#                 return_dict=False
#             )[0]
#             eps1u, _ = torch.split(out1u, 3, dim=1)
#
#             eps1 = eps1u + scale * (eps1 - eps1u)
#
#             # ==============================
#             # 2. ε2 = CFG(xt, prompt2)
#             # ==============================
#             out2 = stage_1.unet(
#                 x,
#                 t,
#                 encoder_hidden_states=prompt2_embeds,
#                 return_dict=False
#             )[0]
#             eps2, _ = torch.split(out2, 3, dim=1)
#
#             out2u = stage_1.unet(
#                 x,
#                 t,
#                 encoder_hidden_states=uncond_embeds,
#                 return_dict=False
#             )[0]
#             eps2u, _ = torch.split(out2u, 3, dim=1)
#
#             eps2 = eps2u + scale * (eps2 - eps2u)
#
#             # ==============================
#             # 3. Hybrid noise = lowpass(eps1) + highpass(eps2)
#             # ==============================
#             # low frequency from eps1
#             low = gaussian_blur(eps1, kernel_size=33, sigma=2)
#
#             # high frequency from eps2
#             high = eps2 - gaussian_blur(eps2, kernel_size=33, sigma=2)
#
#             eps = low + high
#
#             # ==============================
#             # 4. Estimate x0
#             # ==============================
#             x0_hat = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
#
#             # ==============================
#             # 5. DDPM reverse step
#             # ==============================
#             term1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat
#             term2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t) * x
#
#             x_prev = term1 + term2
#
#             x_prev = add_variance(var1, t, x_prev)
#             x = x_prev
#
#             if display and (i % 10 == 0):
#                 tmp = x[0].detach().cpu().permute(1,2,0).numpy()
#                 tmp = tmp/2 + 0.5
#                 tmp = np.clip(tmp, 0, 1)
#                 plt.imshow(tmp)
#                 plt.title(f"Hybrid step {i}, t={t}")
#                 plt.axis("off")
#                 plt.show()
#
#         # final image
#         final = x[0].detach().cpu().permute(1, 2, 0).float().numpy()  # float32
#         final = final / 2 + 0.5
#         final = np.clip(final, 0, 1).astype(np.float32)
#
#         return final
#
#
# prompt1 = prompt_embeds_dict["a minimalist painting of a giant moon over the ocean"].half().to(device)
# prompt2 = prompt_embeds_dict["an intricate ink drawing of a dragon"].half().to(device)
# uncond  = prompt_embeds_dict[""].half().to(device)
#
# noise = torch.randn((1,3,64,64)).half().to(device)
#
# hybrid = make_hybrids(
#     image=noise,
#     i_start=0,
#     prompt1_embeds=prompt1,
#     prompt2_embeds=prompt2,
#     uncond_embeds=uncond,
#     timesteps=strided_timesteps,
#     scale=7,
#     display=False
# )
#
# plt.imshow(hybrid)
# plt.axis("off")
# plt.show()
