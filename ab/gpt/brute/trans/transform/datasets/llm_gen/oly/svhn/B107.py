import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

def transform(norm):
    return T.Compose([T.Pad(4, fill=(118, 93, 71), padding_mode='edge'), T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), interpolation=T.InterpolationMode.BICUBIC, fill=None), T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.5), T.RandomApply([T.GaussianBlur(3, sigma=1.5)], p=0.5), T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[143.357 / 255, 131.548 / 255, 125.276 / 255], std=[68.695 / 255, 65.619 / 255, 67.35 / 255])])