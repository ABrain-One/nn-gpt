import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.8, 1.333)), T.RandomHorizontalFlip(p=0.5), T.RandomApply([T.Solarize(threshold=20)], p=0.2), T.GaussianBlur(kernel_size=3, sigma=1.5, p=0.2), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), T.ToTensor(), T.Normalize(**norm)])