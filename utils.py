import os
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def open_txt(f_path):

    with open(f_path, "r") as f:
        info = f.read().splitlines()
    return info

def write_txt(info, f_path):

    with open(f_path, "w") as f:
        for i in info:
            f.write(f"{i}\n")
    return