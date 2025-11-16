from huggingface_hub import login

token = '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'
login(token=token)

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

device = 'cuda'

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

# Loads your own .pth file here
prompt_embeds_dict = torch.load('prompt_embeds_dict.pth')

# If you want our predefined embeddings, please uncomment the following two lines and run this cell.
# !wget https://cal-cs180.github.io/fa24/hw/proj5/prompt_embeds_dict.pth -O prompt_embeds_dict.pth
# prompt_embeds_dict = torch.load('prompt_embeds_dict.pth')

print("You have the embeddings for the following prompts")
pprint(list(prompt_embeds_dict.keys()))

def seed_everything(seed):
  torch.cuda.manual_seed(seed)
  torch.manual_seed(seed)

YOUR_SEED = 314159
seed_everything(YOUR_SEED)

# Get prompt embeddings from the precomputed cache.
# `prompt_embeds` is of shape [N, 77, 4096]
# 77 comes from the max sequence length that deepfloyd will take
# and 4096 comes from the embedding dimension of the text encoder
# `negative_prompt_embeds` is the same shape as `prompt_embeds` and is used
# for Classifier Free Guidance. You can find out more from:
#   - https://arxiv.org/abs/2207.12598
#   - https://sander.ai/2022/05/26/guidance.html
prompts = [
    # TODO: Choose 3 of your prompts here
'a high quality picture',
'an oil painting of a snowy mountain village',
'a photo of the amalfi coast'
]
prompt_embeds = torch.cat([
    prompt_embeds_dict[prompt] for prompt in prompts
], dim=0)
negative_prompt_embeds = torch.cat(
    [prompt_embeds_dict['']] * len(prompts)
)

# Sample from stage 1
# Outputs a [N, 3, 64, 64] torch tensor
# num_inference_steps is an integer between 1 and 1000, indicating how many
# denoising steps to take: lower is faster, at the cost of reduced quality
stage_1_output = stage_1(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=20,
    output_type="pt"
).images

# Sample from stage 2
# Outputs a [N, 3, 256, 256] torch tensor
# num_inference_steps is an integer between 1 and 1000, indicating how many
# denoising steps to take: lower is faster, at the cost of reduced quality
stage_2_output = stage_2(
    image=stage_1_output,
    num_inference_steps=10,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    output_type="pt",
).images

# Display images
# We need to permute the dimensions because `media.show_images` expects
# a tensor of shape [N, H, W, C], but the above stages gives us tensors of
# shape [N, C, H, W]. We also need to normalize from [-1, 1], which is the
# output of the above stages, to [0, 1]
# media.show_images(
#     stage_1_output.permute(0, 2, 3, 1).cpu() / 2. + 0.5,
#     titles=prompts)
# media.show_images(
#     stage_2_output.permute(0, 2, 3, 1).cpu() / 2. + 0.5,
#     titles=prompts)

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("part0/outputs_10", exist_ok=True)

images = stage_2_output.permute(0, 2, 3, 1).cpu() / 2. + 0.5  # [N,H,W,C], 0~1

for idx, (img_t, prompt) in enumerate(zip(images, prompts)):
    print(f"Image {idx} corresponds to prompt: \"{prompt}\"")

    # convert tensor → numpy
    img = img_t.numpy()

    # 🔧 convert to float32 for matplotlib
    img = img.astype(np.float32)

    # Save
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    filename = f"part0/outputs_10/prompt_{idx}.png"
    pil_img.save(filename)
    print("Saved:", filename)

    # Show
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"Prompt {idx}: {prompt}")
    plt.axis("off")
    plt.show()