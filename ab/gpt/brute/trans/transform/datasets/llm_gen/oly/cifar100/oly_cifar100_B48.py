import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.Resize(128), T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.1), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), T.GaussianBlur(kernel_size=3), T.ToTensor(), T.Normalize(**norm)])