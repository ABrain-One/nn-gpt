import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomResizedCrop(64, scale_range=(0.08, 1.0), ratio_range=(0.75, 1.3333)), T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), T.RandomApply([T.Invert()], p=0.2), T.ToTensor(), T.Normalize(**norm)])