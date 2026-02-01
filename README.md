# Diffusion_Inpainting_OOD

## Overview

Computer Vision project on testing the robustness of SoTA inpainting models in out-of-distribution samples


Image inpainting models are typically evaluated under in-distribution corruption settings, providing limited insight into their robustness under semantic distribution shift. In this work, I study out-of-distribution robustness in image inpainting through controlled perturbations that progressively increase spatial severity and semantic inconsistency. 

In addition to standard variations in mask size, blur, and semantic masking, I introduce a **semantic shuffling** experiment that inserts semantically incompatible facial components into fixed spatial locations. I propose a quantitative boundary measure based on LPIPS perceptual similarity to capture the trade-off between learned priors and local visual evidence. The results show that diffusion-based models exhibit greater robustness to semantic inconsistencies than classical approaches, highlighting the limitations of standard inpainting evaluations and motivating semantically grounded robustness benchmarks.



## Conda Env

We evaluate three pre-trained models : MAE-FAR, RePaint and Stable Diffusion. Since they are evaluated as is, to create the conda environments for each, please follow the source instructions:

- [MAE-FAR](https://github.com/ewrfcas/MAE-FAR)
- [RePaint](https://github.com/andreas128/RePaint)
- Stable Diffusion: Install diffusers and obtain the `runwayml/stable-diffusion-inpainting` model from HuggingFace pretrained models


## Dataset

### Installation

We use CelebAMask-HQ, as it contains image segmentation masks to guide semantic masking and shuffling. To download the dataset, please follow this link: [Reference](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html), and download it into a new directory named `data`.

### Pre-Processing Masks and GT

To create the dataset masks and modified ground truth:

```bash
python -m prep_data \
--gt_img_dir ./data/CelebAMask-HQ/CelebA-HQ-img \
--gt_segm_dir ./data/CelebAMask-HQ/CelebAMask-HQ-mask-anno \
--save_dir ./data/celeba \
--n_sample 500
```

## Evaluating Models

To evaluate each model on the pre-processed dataset, please follow instructions relating to each models. The files that make the models compatible with the desired evaluation are present in `./eval_utils`.

### Model : MAE-FAR

The desired files are found in `./eval_utils/mae_far`. Make sure the source directory ([MAE-FAR](https://github.com/ewrfcas/MAE-FAR)) has been cloned, and do the following before evaluation.

```bash
#Copy config file
cp ./eval_utils/mae_far/config.yml ./MAE-FAR/configs/config.yml

#Copy evaluation script
cp ./eval_utils/mae_far/eval.sh ./MAE-FAR/eval.sh
```

For evaluation, do the following. To change the results directory, or even the mask or ground truth directories, please modify them in the `eval.sh` script.

```bash
#Enter mae_far dir
cd MAE-FAR

#Evaluate
./eval.sh
```


### Model : RePaint

The desired files are found in `./eval_utils/repaint`. Make sure the source directory ([RePaint](https://github.com/andreas128/RePaint)) has been cloned, and do the following before evaluation.

```bash
#Copy config file
cp ./eval_utils/repaint/config.yml ./RePaint/confs/config.yml
```

For evaluation, do the following. To change the results directory, or even the mask or ground truth directories, please modify them in the `config.yml` file.

```bash
python test.py --conf_path confs/config.yml
```

### Model : Stable Diffusion

For stable diffusion, we use the pipeline provided by HuggingFace, which does not require cloning a repository. Therefore, we need to create a directory specific to Stable Diffusion.

```bash
mkdir stable_diff
```

The desired files are found in `./eval_utils/repaint`. 

```bash
#Copy files
cp ./eval_utils/stable_diff/eval.py ./stable_diff/eval.py
```

For evaluation, ensure the working directory is in `stable_diff`.

```bash
#Enter stable_diff dir
cd stable_diff

#Evaluation
python -m eval \
--gt_img_dir /mnt/Diffusion_Inpainting_OOD/data/celeba/images/gt
--gt_mask_dir /mnt/Diffusion_Inpainting_OOD/data/celeba/masks/square/mask_0.8
--save_dir ./results
```



## Evaluation Metrics

The final evaluation metrics comprise of KID, SSIM, and LPIPS. To calculate just the KID metric, remove the `--paired` argument. 

```bash
python -m eval_metrics \
--gt_dir ./data/celeba/images/gt \
--pred_dir ./RePaint/log/square/blur_0/mask_0.8 \
--paired
```

To evaluate the LPIPS and SSIM metrics solely on the masked regions, use the `--mask_dir` argument.

```bash
python -m eval_metrics \
--gt_dir ./data/celeba/images/gt \
--mask_dir ./data/celeba/masks/square/mask_0.8 \
--pred_dir ./RePaint/log/square/blur_0/mask_0.8 \
--paired
```