# -----

# Referenced from https://github.com/Zheng-Chong/CatVTON

# ----


import os
import torch
from cleanfid import fid as FID
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm

from prettytable import PrettyTable
from typing import List, Optional, Tuple, Set

def scan_files_in_dir(directory, postfix: Set[str] = None, progress_bar: tqdm = None) -> list:
    file_list = []
    progress_bar = tqdm(total=0, desc=f"Scanning", ncols=100) if progress_bar is None else progress_bar
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        # elif entry.is_dir():
        #     file_list += scan_files_in_dir(entry.path, postfix=postfix, progress_bar=progress_bar)
    return file_list

class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_folder, mask_folder=None, height=1024):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.mask_folder = mask_folder
        self.height = height
        self.data = self.prepare_data()             #Tuple of (gt_path, pred_path)
        self.to_tensor = transforms.ToTensor()
    
    def extract_id_from_filename(self, filename):

        file_id = filename.split('.')[0]
        return file_id
    
    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={'.jpg', '.png'})
        gt_dict = {self.extract_id_from_filename(file.name): file for file in gt_files}
        pred_files = scan_files_in_dir(self.pred_folder, postfix={'.jpg', '.png'})
        if self.mask_folder is not None:
            print(f"\nConsidering masked regions for evaluation...\n")
            mask_files = scan_files_in_dir(self.mask_folder, postfix=('.jpg', '.png'))
            mask_dict = {self.extract_id_from_filename(file.name): file for file in mask_files}
        
        tuples = []
        for pred_file in pred_files:
            pred_id = self.extract_id_from_filename(pred_file.name)
            if (self.mask_folder is not None) and (pred_id not in mask_dict):
                continue

            if pred_id not in gt_dict:
                print(f"Cannot find gt file for {pred_file}")
            else:
                if self.mask_folder is not None:
                    tuples.append((gt_dict[pred_id].path, pred_file.path, mask_dict[pred_id]))
                else:
                    tuples.append((gt_dict[pred_id].path, pred_file.path, None))
        return tuples
        
    def resize(self, img):
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path, mask_path = self.data[idx]
        gt, pred = self.resize(Image.open(gt_path)), self.resize(Image.open(pred_path))

        if gt.height != self.height:
            gt = self.resize(gt)
        if pred.height != self.height:
            pred = self.resize(pred)
        gt = self.to_tensor(gt)
        pred = self.to_tensor(pred)

        # Apply mask if present
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")  # single channel
            mask = self.resize(mask)
            mask = self.to_tensor(mask)

            # Ensure binary mask (optional but recommended)
            mask = (mask > 0).float()

            gt = gt * mask
            pred = pred * mask

        return gt, pred


def copy_resize_gt(gt_folder, height):
    new_folder = f"{gt_folder}_{height}"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue

        if os.path.isdir(os.path.join(gt_folder, file)):
            print(f"Found dir inside: {os.path.join(gt_folder, file)}. Skipping...")
            continue

        img = Image.open(os.path.join(gt_folder, file))
        w, h = img.size
        new_w = int(w * height / h)
        img = img.resize((new_w, height), Image.LANCZOS)
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        ssim_score += ssim(pred, gt) * batch_size
    return ssim_score / len(dataloader.dataset)


@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to("cuda")
    score = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        # LPIPS needs the images to be in the [-1, 1] range.
        gt = (gt * 2) - 1
        pred = (pred * 2) - 1
        score += lpips_score(gt, pred) * batch_size
    return score / len(dataloader.dataset)


def eval(args):
    # Check gt_folder has images with target height, resize if not
    pred_sample = os.listdir(args.pred_folder)[0]
    gt_sample = os.listdir(args.gt_folder)[0]
    img = Image.open(os.path.join(args.pred_folder, pred_sample))
    gt_img = Image.open(os.path.join(args.gt_folder, gt_sample))
    if img.height != gt_img.height:
        title = "--"*30 + f"Resizing GT Images to height {img.height}" + "--"*30
        print(title)
        args.gt_folder = copy_resize_gt(args.gt_folder, img.height)
        print(f"Number of files in new GT folder: {len(os.listdir(args.gt_folder))}")
        print("-"*len(title))
    
    # Form dataset
    dataset = EvalDataset(args.gt_folder, args.pred_folder, args.mask_folder, img.height)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False
    )
    
    # Calculate Metrics
    header = []
    row = []

    header = ["KID"]
    kid_ = FID.compute_kid(args.gt_folder, args.pred_folder) * 1000
    row = [kid_]

    if args.paired:
        header += ["SSIM", "LPIPS"]
        ssim_ = ssim(dataloader).item()
        lpips_ = lpips(dataloader).item()
        row += [ssim_, lpips_]
    
    # Print Results
    print("GT Folder  : ", args.gt_folder)
    print("Pred Folder: ", args.pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)
    print(table)
    
         
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True, help="GT Image directory")
    parser.add_argument("--pred_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--mask_dir", default=None, help="Use if the (LPIPS, SSIM) should only focus on masked region.")
    parser.add_argument("--paired", action="store_true", help="If true, perceptual metrics (LPIPS, SSIM) will also be evaluated.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    eval(args)