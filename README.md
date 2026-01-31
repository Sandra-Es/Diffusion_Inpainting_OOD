# Diffusion_Inpainting_OOD

## Overview

Computer Vision project on testing the robustness of SoTA inpainting models in out-of-distribution samples


Image inpainting models are typically evaluated under in-distribution corruption settings, providing limited insight into their robustness under semantic distribution shift. In this work, I study out-of-distribution robustness in image inpainting through controlled perturbations that progressively increase spatial severity and semantic inconsistency. 

In addition to standard variations in mask size, blur, and semantic masking, I introduce a **semantic shuffling** experiment that inserts semantically incompatible facial components into fixed spatial locations. I propose a quantitative boundary measure based on LPIPS perceptual similarity to capture the trade-off between learned priors and local visual evidence. The results show that diffusion-based models exhibit greater robustness to semantic inconsistencies than classical approaches, highlighting the limitations of standard inpainting evaluations and motivating semantically grounded robustness benchmarks.



## Conda Env

We test three models : MAE-FAR, RePaint and Stable Diffusion. To create the conda environments for each, please follow the source instructions:

- [MAE-FAR](https://github.com/ewrfcas/MAE-FAR)
- [RePaint](https://github.com/andreas128/RePaint)
- Stable Diffusion: Install diffusers and obtain the `runwayml/stable-diffusion-inpainting` model from HuggingFace pretrained models


## Dataset

### Installation

We use CelebAMask-HQ, as it contains image segmentation masks to guide semantic masking and shuffling. To download the dataset, please follow this link: [Reference](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html).

### Pre-Processing Masks and GT

To create the dataset masks and modified ground truth:

```bash
python -m prep_data \
--gt_img_dir /mnt/data/CelebAMask-HQ/CelebA-HQ-img \
--gt_segm_dir /mnt/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno \
--save_dir /mnt/data/celeba_mod \
--n_sample 500
```

## 
