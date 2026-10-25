import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomRotation(degrees=(-5, 5)), T.RandomHorizontalFlip(p=0.67), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=0.5), T.ToTensor(), T.Normalize(mean=norm[0], std=norm[1])])