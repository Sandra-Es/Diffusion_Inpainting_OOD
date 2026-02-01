import inspect
from typing import List, Optional, Union
from argparse import ArgumentParser
import os

import numpy as np
import torch
from tqdm import tqdm

import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline



def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--gt_img_dir", type=str, required=True)
    parser.add_argument("--gt_mask_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    #image, mask dirs
    img_dir = args.gt_img_dir
    mask_root_dir = args.gt_mask_dir
    save_dir = args.save_dir

    #Load model
    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    #Enforce no text-based guidance
    prompt = ""
    guidance_scale = 0.0
    num_samples = 1

    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.endswith(".jpg"):
            continue

        img = PIL.Image.open(os.path.join(img_dir, img_name)).convert("RGB")

        for mask_type in os.listdir(mask_root_dir):

            save_path = os.path.join(save_dir, mask_type, img_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                continue

            mask_path = os.path.join(mask_root_dir, mask_type, img_name.replace(".jpg", ".png"))
            if not os.path.exists(mask_path):
                continue
            mask = PIL.Image.open(mask_path).convert("RGB")


            #Inference
            output = pipe(prompt=prompt, 
                            image=img, mask_image=mask, 
                            num_inference_steps=50, 
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=num_samples).images
            inpainted_image = output[0]

            #Save output
            inpainted_image.save(save_path)



if __name__ == "__main__":
    main()