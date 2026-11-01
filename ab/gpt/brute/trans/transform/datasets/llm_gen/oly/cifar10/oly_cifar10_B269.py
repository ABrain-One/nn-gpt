import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.Resize(64), T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)), T.RandomCrop(32), T.Resize(64), T.ToTensor(), T.Normalize(**norm)])