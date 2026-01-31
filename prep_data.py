import os
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Dict

from utils import *



def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--gt_img_dir", type=str, required=True)
    parser.add_argument("--gt_segm_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def extract_segm_masks(img_id, mask_dir):

    img_id_pad = img_id.zfill(5)
    
    masks = {"l_eye": None,
             "r_eye": None,
             "nose": None,
             "mouth": None,
             "l_ear": None,
             "r_ear": None,}
    
    mask_names = {"l_eye": "id_l_eye.png",
                  "r_eye": "id_r_eye.png",
                  "nose": "id_nose.png",
                  "mouth": "id_mouth.png",
                  "l_ear": "id_l_ear.png",
                  "r_ear": "id_r_ear.png",
                  "u_lip": "id_u_lip.png",
                  "l_lip": "id_l_lip.png",}

    for sub in os.listdir(mask_dir):
        sub_path = os.path.join(mask_dir, sub)

        for mask_file in os.listdir(sub_path):

            mask_id = mask_file.split("_")[0]
            if img_id_pad == mask_id:

                for id in masks.keys():

                    if id in ["u_lip", "l_lip"]:
                        continue

                    if id == "mouth":
                        mouth_name = mask_names["mouth"].replace("id", mask_id)
                        u_lip_name = mask_names["u_lip"].replace("id", mask_id)
                        l_lip_name = mask_names["l_lip"].replace("id", mask_id)

                        mouth_mask = np.zeros((512, 512), dtype=np.uint8)
                        u_lip_mask = np.zeros((512, 512), dtype=np.uint8)
                        l_lip_mask = np.zeros((512, 512), dtype=np.uint8)

                        if os.path.exists(os.path.join(sub_path, mouth_name)):
                            mouth_mask = cv2.imread(os.path.join(sub_path, mouth_name), cv2.IMREAD_GRAYSCALE)
                        if os.path.exists(os.path.join(sub_path, u_lip_name)):
                            u_lip_mask = cv2.imread(os.path.join(sub_path, u_lip_name), cv2.IMREAD_GRAYSCALE)
                        if os.path.exists(os.path.join(sub_path, l_lip_name)):
                            l_lip_mask = cv2.imread(os.path.join(sub_path, l_lip_name), cv2.IMREAD_GRAYSCALE)

                        mask = mouth_mask + u_lip_mask + l_lip_mask
                        mask[mask > 0] = 255
                        masks[id] = mask
                        continue

                    mask_name = mask_names[id].replace("id", mask_id)
                    mask_path = os.path.join(sub_path, mask_name)

                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        masks[id] = mask

    return masks

def conv_to_bbox(bin_mask, active_percent=1.0):
    mask = bin_mask.copy()
    
    ys, xs = np.where(mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    mask[:, :] = 0

    if active_percent == 1.0:
        mask[y_min:y_max + 1, x_min:x_max + 1] = 1
    else:
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        new_w = int(w * active_percent)
        new_h = int(h * active_percent)

        x_c = x_min + w // 2
        y_c = y_min + h // 2

        x_min_new = x_c - new_w // 2
        x_max_new = x_min + new_w - 1

        y_min_new = y_c - new_h // 2
        y_max_new = y_min + new_h - 1

        mask[y_min_new:y_max_new + 1, x_min_new:x_max_new + 1] = 1
    return mask

def save_masks_occlussion(imgs_list: List, 
                            gt_mask_dir: str, save_mask_dir: str, 
                            mask_percs: List = [0.8, 1.0, 1.5, 2.0]):

    print(f"\n\n ----- Generating Masks : Occlusion ---- \n\n")

    for mask_perc in mask_percs:

        print("Mask Percent:", mask_perc)
        curr_save_dir = os.path.join(save_mask_dir, f"mask_{mask_perc}")
        os.makedirs(curr_save_dir, exist_ok=True)

        for img_name in tqdm(imgs_list):
            #Get segmentation masks
            img_id = img_name.split(".")[0]
            segm_masks = extract_segm_masks(img_id, gt_mask_dir)

            #Convert to square bbox mask
            bbox_mask = conv_to_bbox(segm_masks["nose"], active_percent=mask_perc)

            #Save mask
            save_path = os.path.join(curr_save_dir, img_name.replace(".jpg", ".png"))
            cv2.imwrite(save_path, bbox_mask * 255)

    print(f"\n\n ---- Done Generating Masks : Occlusion ---- \n\n")

def save_gt_blur(imgs_list: List, 
                    gt_img_dir: str, save_gt_dir: str,
                    k_sizes: List = [13, 25, 37, 51]):

    print(f"\n\n ---- Generating GT : Blur ---- \n\n")

    for k in [13, 25, 37, 51]:

        print("Kernel Size:", k)
        blur_img_dir_k = os.path.join(save_gt_dir, f"blur_{k}")
        os.makedirs(blur_img_dir_k, exist_ok=True)

        for img_name in tqdm(imgs_list):
            img = cv2.imread(os.path.join(gt_img_dir, img_name))
            img_trim = cv2.resize(img, (512, 512))

            img_blur = cv2.GaussianBlur(img_trim, (k, k), 0)

            save_path = os.path.join(blur_img_dir_k, img_name)
            cv2.imwrite(save_path, img_blur)

    print(f"\n\n ---- Done Generating GT : Blur ---- \n\n")

def save_masks_sem(imgs_list: List, 
                    gt_mask_dir: str, save_mask_dir: str,
                    sem_cats = ["mouth", "nose", "l_eye", "both_eye"]):

    print(f"\n\n ---- Generating Masks: Semantic ---- \n\n")

    for mask_type in ["mouth", "nose", "l_eye", "both_eye"]:

        print("Mask Type:", mask_type)
        save_dir = os.path.join(save_mask_dir, mask_type)
        os.makedirs(save_dir, exist_ok=True)

        for img_name in tqdm(imgs_list):

            save_path = os.path.join(save_dir, img_name.replace(".jpg", ".png"))
            if os.path.exists(save_path):
                continue

            #Get segmentation masks
            img_id = img_name.split(".")[0]
            segm_masks = extract_segm_masks(img_id, gt_mask_dir)

            #Convert to square bbox mask
            try:
                if mask_type == "both_eye":
                    r_mask = conv_to_bbox(segm_masks["r_eye"], active_percent=1.5)
                    l_mask = conv_to_bbox(segm_masks["l_eye"], active_percent=1.5)
                    combined_mask = np.maximum(r_mask, l_mask)
                    final_mask = combined_mask
                else:
                    final_mask = conv_to_bbox(segm_masks[mask_type], active_percent=1.0)
            except:
                print("Error in:", img_name)
                continue

            #Save mask
            cv2.imwrite(save_path, final_mask * 255)

    print(f"\n\n ---- Done Generating Masks: Semantic ---- \n\n")

def save_gt_masks_sem_shuffle(imgs_list: List, 
                                gt_img_dir: str, save_gt_dir: str,
                                gt_mask_dir: str, save_mask_dir: str,
                                parts_swap = [["r_eye", "mouth"]],
                                mask_percs = [0.2, 0.4, 0.6, 0.8, 1.0]
                                ):

    parts_swap = [["r_eye", "mouth"]]
    parts_name = ["_".join(parts_swap)]

    #Create save dirs if not exist
    os.makedirs(save_gt_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    print(f"\n\n ---- Generating GT and Masks : Semantic Shuffle ---- \n\n")

    for swap, name in zip(parts_swap, parts_name):

        for img_name in tqdm(imgs_list):

            save_path = os.path.join(save_gt_dir, name, img_name)
            if os.path.exists(save_path):
                continue

            #Read image and trim
            img_path = os.path.join(gt_img_dir, img_name)
            img = cv2.imread(img_path)

            segm_masks = extract_segm_masks(img_name.replace(".jpg", ""), mask_dir)

            img_trim = cv2.resize(img, (512, 512))

            #Swap parts
            try:
                img_swap, bin_mask = overlay_swap_with_smoothing(img_trim, 
                                                                    segm_masks[swap[0]], 
                                                                    segm_masks[swap[1]])
            except:
                continue

            #Save image
            save_path = os.path.join(save_gt_dir, name, img_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_swap[..., ::])

            for mask_perc in mask_percs:
                #Convert to square bbox mask
                bbox_mask = conv_to_bbox(bin_mask, active_percent=mask_perc)

                #Save mask
                save_path = os.path.join(save_mask_dir, name, f"mask_{mask_perc}", img_name.replace(".jpg", ".png"))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, bbox_mask * 255)



def main():

    args = parser_args()

    #Create save dir
    os.makedirs(args.save_dir, exist_ok=True)

    #Get list of 500 images if exists
    imgs_list_path = os.path.join(args.save_dir, "images-500.txt")
    if os.path.exists(imgs_list_path):
        imgs_list = open_txt(imgs_list_path)
    else:
        img_samples = random.sample(os.listdir(args.gt_img_dir, 500))
        write_txt(img_samples, imgs_list_path)

    
    #Save Maks : Occlusion
    save_masks_occlussion(imgs_list = imgs_list, 
                            save_mask_dir = os.path.join(args.save_dir, "masks", "square"))

    #Save GT : Blur
    save_gt_blur(imgs_list = imgs_list,
                    save_gt_dir = os.path.join(args.save_dir, "images", "blur"))


    #Save Masks : Semantic
    save_masks_sem(imgs_list = imgs_list,
                    save_mask_dir = os.path.join(args.save_dir, "masks", "semantic"))


    #Save Masks and GT : Semantic Shuffle
    save_gt_masks_sem_shuffle(imgs_list = imgs_list,
                                gt_img_dir = args.gt_img_dir,
                                save_gt_dir = os.path.join(args.save_dir, "images", "semantic_swap"),
                                gt_mask_dir = args.gt_segm_dir,
                                save_mask_dir = os.path.join(args.save_dir, "masks", "semantic_swap")
                                )




if __name__ == "__main__":
    main()