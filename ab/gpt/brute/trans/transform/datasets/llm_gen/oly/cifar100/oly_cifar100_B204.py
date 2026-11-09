import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomResizedCrop(64, scale=(0.75, 1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.GaussianBlur(kernel_size=3, sigma=0.5), T.ToTensor(), T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])