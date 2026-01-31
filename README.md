# Diffusion_Inpainting_OOD

## Overview

Computer Vision project on testing the robustness of SoTA inpainting models in out-of-distribution samples


Image inpainting models are typically evaluated under in-distribution corruption settings, providing limited insight into their robustness under semantic distribution shift. In this work, I study out-of-distribution robustness in image inpainting through controlled perturbations that progressively increase spatial severity and semantic inconsistency. 

In addition to standard variations in mask size, blur, and semantic masking, I introduce a **semantic shuffling** experiment that inserts semantically incompatible facial components into fixed spatial locations. I propose a quantitative boundary measure based on LPIPS perceptual similarity to capture the trade-off between learned priors and local visual evidence. The results show that diffusion-based models exhibit greater robustness to semantic inconsistencies than classical approaches, highlighting the limitations of standard inpainting evaluations and motivating semantically grounded robustness benchmarks.

